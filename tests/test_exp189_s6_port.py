"""CPU checks; unittest here also allows pytest collection on the GPU host.

v1.1.2 (refactor lead, 2026-10-11): generate kernel-spec fixtures in a private
temporary directory, independent of staged experiment version directories.
v1.1.1 (astra, 2026-10-11): use the tools package without changing sys.path;
collection must not shadow training modules in other tests or spawned workers.
v1.1.0 (Claude opus for B, 2026-10-11): code-tree / closure / wheel pins, bounded
wheel install, process-group runner, anchor timing and anatomy-fallback counting.
"""
import ast
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import exp189_s6_port as s6

PUBLIC_MODULES = ("native_cache.py", "native_data.py", "input_provenance.py", "intensity_mapping.py",
                  "coronal_views.py", "exact_duplicates.py", "dicom_geometry.py", "common.py")


def _fake_dataset(root: Path) -> dict:
    """A miniature code dataset mount: cloud_code/training tree, anatomy.pt, two wheels."""
    training = root / "cloud_code" / "training"
    (training / "vendor_runtime").mkdir(parents=True)
    (root.parent / "work").mkdir(exist_ok=True)
    (training / "__init__.py").write_text("# pkg\n", encoding="utf-8")
    for name in PUBLIC_MODULES:
        (training / name).write_text(f"# public module {name}\nfrom . import common\n", encoding="utf-8")
    (training / "geometry_bank.py").write_text("from training.geometry_runtime import grid\n", encoding="utf-8")
    (training / "geometry_runtime.py").write_text("grid = 1\n", encoding="utf-8")
    (training / "anatomy.py").write_text("import json\nfrom .vendor_runtime import helper\n", encoding="utf-8")
    (training / "anatomy_runtime.py").write_text("def native_volume(p):\n    return p\n", encoding="utf-8")
    (training / "vendor_runtime" / "__init__.py").write_text("", encoding="utf-8")
    (training / "vendor_runtime" / "helper.py").write_text("X = 2\n", encoding="utf-8")
    (training / "unused.py").write_text("import os\n", encoding="utf-8")
    (root / "anatomy.pt").write_bytes(b"anatomy-test")
    wheels = root / "dependencies" / "wheels"
    wheels.mkdir(parents=True)
    for name in s6.WHEEL_HASHES:
        (wheels / name).write_bytes(b"wheel:" + name.encode())
    return {
        "code": {name: s6.sha256(training / name) for name in PUBLIC_MODULES},
        "anatomy": s6.sha256(root / "anatomy.pt"),
        "wheels": {name: s6.sha256(wheels / name) for name in s6.WHEEL_HASHES},
    }


class _FakePopen:
    instances = []

    def __init__(self, args, **kwargs):
        self.args, self.kwargs, self.pid, self.returncode = args, kwargs, 4242, None
        _FakePopen.instances.append(self)

    def communicate(self):
        raise KeyboardInterrupt('external cancellation')

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        self.returncode = -9 if self.returncode is None else self.returncode
        return self.returncode


class S6PortTests(unittest.TestCase):
    def test_cpu_parity_accepts_public_exit_map_and_rejects_cuda_or_failed_worker(self):
        runner = s6.HERE.parent / 'artifacts/experiments/EXP189_S6_PORT/v1.2.0/cpu_parity.py'
        tree = ast.parse(runner.read_text(encoding='utf-8-sig'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'cpu_worker_evidence')
        scope = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(runner), 'exec'), scope)
        check = scope['cpu_worker_evidence']
        worker = {'device': 'cpu', 'auxiliary_device': 'cpu', 'cuda_initialized': False}
        receipt = {'workers': [worker], 'worker_exit_codes': {'0': 0}}
        self.assertEqual(check([receipt, receipt]), (True, True))
        self.assertEqual(check([{**receipt, 'worker_exit_codes': {'0': 1}}]), (True, False))
        self.assertEqual(check([{**receipt, 'worker_exit_codes': [0]}]), (True, False))
        self.assertEqual(check([{**receipt, 'workers': [{**worker, 'auxiliary_device': 'cuda'}]}]),
                         (False, True))
        self.assertEqual(check([{**receipt, 'workers': [{**worker, 'cuda_initialized': True}]}]),
                         (False, True))
        self.assertEqual(check([{'workers': [], 'worker_exit_codes': {}}]), (False, False))

    def test_cpu_anatomy_and_side_models_never_probe_or_select_cuda(self):
        """Exercise both real setup bodies even when CUDA availability is unsafe."""
        from tools import exp189_make_kernels as generator

        original = {name: (s6.SOURCE / name).read_text(encoding="utf-8")
                    for name in s6.SOURCE_HASHES}
        with tempfile.TemporaryDirectory() as tmp:
            runtime = s6._diagnostic_source(s6.SOURCE, Path(tmp))
            sources = [(runtime / 'compact_infer.py').read_text(encoding='utf-8'),
                       generator.reference_runtime_sources(original, cpu=True)['compact_infer.py']]
        for source in sources:
            with self.subTest(adapter='port' if 'self.device' in source else 'reference'):
                tree = ast.parse(source)
                cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CStudies')
                setup = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_setup')
                devices = []

                class SideNet:
                    def load_state_dict(self, state, strict):
                        self_outer.assertTrue(strict)

                    def eval(self):
                        return self

                    def to(self, device):
                        devices.append(('side', device))
                        return self

                self_outer = self
                training = types.ModuleType('training')
                training.__path__ = []
                training.geometry_bank = types.ModuleType('training.geometry_bank')
                training.dicom_geometry = types.SimpleNamespace(_forced_side_wrapped=True)
                modules = {
                    'training': training, 'training.geometry_bank': training.geometry_bank,
                    'training.dicom_geometry': training.dicom_geometry,
                    'probe_infer': types.SimpleNamespace(install_native=lambda _: object()),
                    'px_cache': types.SimpleNamespace(StudySlices=lambda *args: object()),
                    'compact_train': types.SimpleNamespace(window_rows=object()),
                    'knee_anchor_bank_v2': types.SimpleNamespace(
                        load_model=lambda path, device: devices.append(('anatomy', device)),
                        AnchorBank=lambda model, device: devices.append(('bank', device))),
                    'side_common': types.SimpleNamespace(build_side_net=SideNet),
                }
                def forbidden_cuda():
                    raise AssertionError('CPU setup queried CUDA availability')

                torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=forbidden_cuda),
                    set_num_threads=lambda _: None, load=lambda *args, **kwargs: {})
                scope = {'sys': types.SimpleNamespace(path=[]), 'HERE': Path('.'), 'torch': torch}
                exec(compile(ast.Module(body=[setup], type_ignores=[]), '<real-setup>', 'exec'), scope)
                dataset = types.SimpleNamespace(code='code', tmp='tmp', size=384, views='v4',
                    anchor_weights='anatomy.pt', side_model='side.pt', device='cpu')
                with patch.dict(sys.modules, modules):
                    scope['_setup'](dataset)
                self.assertEqual(devices, [('anatomy', 'cpu'), ('bank', 'cpu'), ('side', 'cpu')])
                self.assertEqual(dataset.side_dev, 'cpu')

    def test_runtime_cpu_policy_and_completion_fields(self):
        """The executable derivative cannot retain the removed deadline state."""
        with tempfile.TemporaryDirectory() as d:
            runtime = s6._diagnostic_source(s6.SOURCE, Path(d))
            worker = (runtime / "compact_infer.py").read_text(encoding="utf-8")
            controller = (runtime / "probe_dual.py").read_text(encoding="utf-8")
        for source in (worker, controller):
            compile(source, "<s6-runtime>", "exec")
            for forbidden in ("timed_out", "a.deadline", "a.item_timeout", '"--deadline"'):
                self.assertNotIn(forbidden, source)
        self.assertIn('env["CUDA_VISIBLE_DEVICES"] = "-1"',
                      Path(s6.__file__).read_text(encoding="utf-8"))
        self.assertIn('names, visible = ["CPU"], ["-1"]', controller)
        self.assertIn('CUDA_VISIBLE_DEVICES=visible[gi]', controller)
        self.assertIn('amp = "fp32" if dev.type == "cpu"', worker)
        self.assertIn('("studies", "scored", "device", "seconds", "amp")', worker)

    def test_public_reference_authentication_and_unlimited_cpu_derivative(self):
        from tools import exp189_make_kernels as generator

        public = json.loads(generator.NOTEBOOK.read_text(encoding="utf-8"))
        generator.public_reference_cells(public)
        public["cells"][5]["source"] = "".join(public["cells"][5]["source"]) + "\n# tampered\n"
        with self.assertRaisesRegex(ValueError, "public cell 5 differs"):
            generator.public_reference_cells(public)
        original = {name: (s6.SOURCE / name).read_text(encoding="utf-8")
                    for name in s6.SOURCE_HASHES}
        derived = generator.reference_runtime_sources(original, cpu=True)
        for source in (derived["compact_infer.py"], derived["probe_dual.py"]):
            for forbidden in ("timed_out", "a.deadline", "a.item_timeout", '"--deadline"'):
                self.assertNotIn(forbidden, source)
        self.assertIn('dev = torch.device("cpu")', derived["compact_infer.py"])
        self.assertIn('CUDA_VISIBLE_DEVICES="-1"', derived["probe_dual.py"])

    def test_worker_resolves_competition_series_tree_and_retains_traceback(self):
        """The real CStudies entry sees staged metadata and series DICOM paths."""
        import numpy as np

        class Tensor:
            def __init__(self, array):
                self.array = array

            def numpy(self):
                return self.array

            @property
            def shape(self):
                return self.array.shape

        # CStudies' CPU path needs only these tensor constructors. Keeping the
        # model stub local lets the regression run in the project's CPU venv.
        torch = types.SimpleNamespace(
            utils=types.SimpleNamespace(data=types.SimpleNamespace(Dataset=object)),
            from_numpy=lambda array: Tensor(array),
            zeros=lambda shape, dtype: Tensor(np.zeros(shape, dtype=dtype)),
            uint8=np.uint8, float32=np.float32,
        )

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            uid, sid = "study-001", "series-001"
            dicom = root / "competition" / "test_series" / uid / sid / "image.dcm"
            dicom.parent.mkdir(parents=True)
            dicom.write_bytes(b"synthetic-dicom")
            with (root / "competition" / "test_series.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([s6.UID, "SeriesInstanceUID", "Anatomical_Plane", "SeriesDescription"])
                writer.writerow([uid, sid, "Sagittal", "synthetic"])
            stage = root / "stage"
            stage.mkdir()
            # A directory copy gives Windows the same study/series tree without
            # requiring the symlink privilege needed only by production staging.
            with patch.object(s6.os, "symlink", side_effect=lambda src, dst, **kw: shutil.copytree(src, dst)):
                s6._stage_studies([dicom.parent.parent], [uid], stage)
            runtime = s6._diagnostic_source(s6.SOURCE, root)
            spec = importlib.util.spec_from_file_location("exp189_runtime_worker_test", runtime / "compact_infer.py")
            worker = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {"torch": torch}):
                spec.loader.exec_module(worker)

            class Native:
                fail = False

                def prepare(self, data, work, *, split, study_ids, source_directory):
                    self_outer.assertEqual((split, study_ids, source_directory), ("test", [uid], True))
                    with (Path(data) / "test_series.csv").open(newline="", encoding="utf-8") as f:
                        rows = list(csv.DictReader(f))
                    self_outer.assertEqual(rows[0]["SeriesInstanceUID"], sid)
                    self_outer.assertEqual(rows[0]["Anatomical_Plane"], "Sagittal")
                    path = Path(data) / "test_series" / uid / sid / "image.dcm"
                    self_outer.assertEqual(path.read_bytes(), b"synthetic-dicom")
                    if self.fail:
                        raise FileNotFoundError(str(path / "missing-after-read"))

            class Builder:
                cache = ""

                def build(self, study):
                    return {"px": torch.zeros((3, 4, 4), dtype=torch.uint8),
                            "meta": torch.zeros((3, 8), dtype=torch.float32), "blocks": []}

            native = Native()
            self_outer = self

            def fake_setup(instance):
                instance.native = native
                instance.PI = types.SimpleNamespace(OneStudySource=types.SimpleNamespace(current=None))
                instance.GB = None
                instance.bank = None
                instance.side_net = None
                instance.builder = Builder()
                instance.window_rows = lambda *args, **kwargs: [(0, 1, 0, 1, 2)]

            with patch.object(worker.CStudies, "_setup", fake_setup):
                ds = worker.CStudies(stage, "test", [uid], root / "scratch", root, 4)
                result = ds[0]
                self.assertIsNone(result["error"])
                self.assertEqual(result["images"].shape[0], 1)
                native.fail = True
                failed = ds[0]
                self.assertIn("missing-after-read", failed["error"])
                self.assertIn("FileNotFoundError", failed["traceback"])
                self.assertIn("prepare", failed["traceback"])

            controller = (runtime / "probe_dual.py").read_text(encoding="utf-8")
            self.assertIn("for u in missing}", controller)
            self.assertNotIn("missing[:40]", controller)
            source = (runtime / "compact_infer.py").read_text(encoding="utf-8")
            self.assertIn('"per_study": per_study,', source)
            self.assertIn('item.get("traceback", "")', source)

    def test_manifest_model_roster_accepts_exact_four_and_rejects_extra(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d)
            manifest = {
                "bundle_id": s6.BUNDLE_ID, "arm": "E_lrprior",
                "models": [f"E_lrprior_s0_f{i}.pt" for i in range(4)],
                "backbone": "convnext_small", "size": 384, "crop": 352,
                "slices": 3, "phys_mm": 3.5, "views": "v4", "eval_stride": 1,
                "state_spec": "state_spec.json", "side_model": "side_cls.pt",
                "side_threshold": 2.0, "side_position_mm": 20.0, "side_max_x": 0.0,
                "gold_in_training": 0, "official_labels_read_by_trainer": False,
                "anchor_weights_sha256": s6.ANATOMY_SHA256, "sha256": s6.HASHES,
            }
            (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (bundle / "README.md").write_text(s6.README_PERMISSION, encoding="utf-8")
            for name in manifest["models"]:
                (bundle / name).write_bytes(b"test-only")
            with patch.object(s6, "_check_hash"):
                self.assertEqual(s6.verify_bundle(bundle)["models"], manifest["models"])
                (bundle / "E_lrprior_s9_f0.pt").write_bytes(b"unexpected")
                with self.assertRaisesRegex(ValueError, "model roster"):
                    s6.verify_bundle(bundle)

    def test_controller_waits_until_completion(self):
        s6._run_controller([sys.executable, "-c", "import time; time.sleep(.2)"], dict(os.environ))

    def test_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "weight.pt"
            file.write_bytes(b"correct")
            expected = s6.sha256(file)
            s6._check_hash(file, expected)
            file.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                s6._check_hash(file, expected)

    def test_p954_per_finding_average_ranks_and_ties(self):
        ids = ["C", "A", "B"]
        r = [{s6.UID: uid, **{label: [0.1, 0.1, 0.9][i] for label in s6.LABELS}}
             for i, uid in enumerate(ids)]
        s = [{s6.UID: uid, **{label: [0.8, 0.2, 0.4][i] for label in s6.LABELS}}
             for i, uid in enumerate(ids)]
        # Independent expected pandas average ranks: R=(.5,.5,1), S=(1,1/3,2/3).
        expected = [.7 * .5 + .3 * 1, .7 * .5 + .3 / 3, .7 + .3 * 2 / 3]
        expected = [struct.unpack("f", struct.pack("f", value))[0] for value in expected]
        result = s6.p954(r, s)
        self.assertEqual([row[s6.UID] for row in result], ids)
        self.assertEqual([row["ACL"] for row in result], expected)
        self.assertTrue(all(result[i][label] == expected[i]
                            for i in range(3) for label in s6.LABELS))
        # A second finding reverses the Raptor order independently of ACL.
        for i, row in enumerate(r):
            row["MCL"] = [0.9, 0.4, 0.1][i]
        mcl = s6.p954(r, s)
        mcl_expected = [.7 + .3, .7 * (2 / 3) + .3 / 3, .7 / 3 + .3 * (2 / 3)]
        mcl_expected = [struct.unpack("f", struct.pack("f", value))[0] for value in mcl_expected]
        self.assertEqual([row["MCL"] for row in mcl], mcl_expected)
        self.assertEqual([row["ACL"] for row in mcl], expected)
        with self.assertRaisesRegex(ValueError, "identical"):
            s6.p954(r, list(reversed(s)))

    def test_output_schema_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([s6.UID, *s6.LABELS])
                for uid in ["z", "a"]:
                    w.writerow([uid, *(["0.123456"] * 12)])
            rows = s6.read_predictions(path, ["z", "a"])
            self.assertEqual(len(rows), 2)
            self.assertEqual(list(rows[0]), [s6.UID, *s6.LABELS])
            self.assertEqual(rows[0]["ACL"], 0.123456)
            with self.assertRaisesRegex(ValueError, "order"):
                s6.read_predictions(path, ["a", "z"])
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([s6.UID, *reversed(s6.LABELS)])
                w.writerow(["z", *([0.5] * 12)])
            with self.assertRaisesRegex(ValueError, "columns"):
                s6.read_predictions(path, ["z"])

    # ------------------------------------------------------------ v1.1.0 pins

    def _pins(self, hashes):
        return [patch.object(s6, "CODE_HASHES", hashes["code"]),
                patch.object(s6, "ANATOMY_SHA256", hashes["anatomy"]),
                patch.object(s6, "WHEEL_HASHES", hashes["wheels"])]

    def test_unpopulated_code_tree_pin_fails_closed(self):
        # State-independent: simulate the unpinned source; the real pin is never edited.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "mount"
            hashes = _fake_dataset(root)
            with self._pins(hashes)[0], self._pins(hashes)[1], \
                    patch.object(s6, "CODE_TREE_SHA256", None), patch.object(s6, "CODE_TREE_FILES", None):
                with self.assertRaisesRegex(ValueError, "not populated; fail closed"):
                    s6.verify_code(root / "cloud_code", root / "anatomy.pt")

    def test_code_tree_closure_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            root, work = Path(d) / "mount", Path(d) / "work"
            hashes = _fake_dataset(root)
            p1, p2, p3 = self._pins(hashes)
            with p1, p2, p3:
                observed = s6.observe_code(root, work / "observed.json", work_dir=work)
                self.assertTrue(observed["wheel_pins_match"] and observed["public_module_pins_match"]
                                and observed["direct_modules_present"])
                closure = observed["closure_sha256"]
                # Static closure follows absolute and relative imports into vendored packages.
                for rel in ("training/__init__.py", "training/geometry_bank.py", "training/geometry_runtime.py",
                            "training/anatomy.py", "training/anatomy_runtime.py",
                            "training/vendor_runtime/__init__.py", "training/vendor_runtime/helper.py",
                            "training/common.py"):
                    self.assertIn(rel, closure)
                self.assertNotIn("training/unused.py", closure)
                digest = observed["tree_sha256"]
                code, anatomy = s6.prepare_code(root, work / "prep", install_wheels=False,
                                                code_tree_sha256=digest)
                record = s6.verify_code(code, anatomy, tree_sha256=digest)
                self.assertEqual(record["closure_sha256"], closure)
                # Pinned in source: same tree passes without an explicit digest.
                with patch.object(s6, "CODE_TREE_SHA256", digest), patch.object(s6, "CODE_TREE_FILES", observed["files"]):
                    s6.verify_code(code, anatomy)
                    # One changed byte in a transitively executed module fails closed.
                    helper = code / "training/vendor_runtime/helper.py"
                    helper.write_text("X = 3\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "code tree differs"):
                        s6.verify_code(code, anatomy)
                    helper.write_text("X = 2\n", encoding="utf-8")
                    s6.verify_code(code, anatomy)
                    # An extra (e.g. planted .pyc or shadowing module) file also fails closed.
                    (code / "training" / "json.py").write_text("", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "code tree differs"):
                        s6.verify_code(code, anatomy)
                    (code / "training" / "json.py").unlink()
                    # A missing directly imported module is rejected even if a digest matched.
                    (code / "training/geometry_bank.py").unlink()
                    gone = s6.code_tree_manifest(code)["tree_sha256"]
                    with self.assertRaisesRegex(ValueError, "closure is incomplete"):
                        s6.verify_code(code, anatomy, tree_sha256=gone)
                # Destination must be fresh.
                with self.assertRaises(FileExistsError):
                    s6.prepare_code(root, work / "prep", install_wheels=False, code_tree_sha256=digest)

    def test_wheel_roster_and_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root, work = Path(d) / "mount", Path(d) / "work"
            hashes = _fake_dataset(root)
            p1, p2, p3 = self._pins(hashes)
            with p1, p2, p3:
                digest = s6.observe_code(root, work / "o.json", work_dir=work)["tree_sha256"]
                wheel = root / "dependencies/wheels/timm-1.0.25-py3-none-any.whl"
                wheel.write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    s6.prepare_code(root, work / "a", install_wheels=False, code_tree_sha256=digest)
                wheel.write_bytes(b"wheel:timm-1.0.25-py3-none-any.whl")
                (root / "dependencies/wheels/extra-1.0-py3-none-any.whl").write_bytes(b"x")
                with self.assertRaisesRegex(ValueError, "wheel roster"):
                    s6.prepare_code(root, work / "b", install_wheels=False, code_tree_sha256=digest)

    def test_pinned_wheel_digests_are_the_pypi_values(self):
        self.assertEqual(s6.WHEEL_HASHES, {
            "pydicom-3.0.1-py3-none-any.whl": "db32f78b2641bd7972096b8289111ddab01fb221610de8d7afa835eb938adb41",
            "timm-1.0.25-py3-none-any.whl": "bef7f61dd717cb2dbbb7e326f143e13d660a47ecbd84116e6fe33732bed5c484"})
        self.assertEqual(s6.WHEEL_VERSIONS, {"pydicom": "3.0.1", "timm": "1.0.25"})

    def test_pin_code_validates_observation_and_writes_once(self):
        with tempfile.TemporaryDirectory() as d:
            root, work = Path(d) / "mount", Path(d) / "work"
            hashes = _fake_dataset(root)
            p1, p2, p3 = self._pins(hashes)
            with p1, p2, p3:
                observed = s6.observe_code(root, work / "o.json", work_dir=work)
                # Temporary unpinned copy of the port source (works whether or not the real
                # source is already pinned); the real tools/exp189_s6_port.py is never written.
                copy = work / "port_copy.py"
                source = Path(s6.__file__).read_text(encoding="utf-8")
                source = re.sub(r'^CODE_TREE_SHA256: str \| None = .*$', "CODE_TREE_SHA256: str | None = None",
                                source, count=1, flags=re.M)
                source = re.sub(r'^CODE_TREE_FILES: int \| None = .*$', "CODE_TREE_FILES: int | None = None",
                                source, count=1, flags=re.M)
                copy.write_text(source, encoding="utf-8", newline=chr(10))
                real_before = s6.sha256(Path(s6.__file__))
                pin = s6.pin_code(observed, write=True, path=copy)
                self.assertEqual(s6.sha256(Path(s6.__file__)), real_before)
                text = copy.read_text(encoding="utf-8")
                self.assertIn(f'CODE_TREE_SHA256: str | None = "{observed["tree_sha256"]}"\n', text)
                self.assertIn(f"CODE_TREE_FILES: int | None = {observed['files']}\n", text)
                self.assertEqual(pin["port_sha256"], s6.sha256(copy))
                with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                    s6.pin_code(observed, write=True, path=copy)
                bad = dict(observed, wheel_sha256={"x.whl": "0" * 64})
                with self.assertRaisesRegex(ValueError, "wheels differ"):
                    s6.pin_code(bad)
                forged = dict(observed, tree_sha256="0" * 64)
                with self.assertRaisesRegex(ValueError, "inconsistent"):
                    s6.pin_code(forged)
            # Real pins: the fake tree's public modules do not match heliosli's hashes.
            with self.assertRaisesRegex(ValueError, "public mainline module pins"):
                s6.check_observation(observed)

    def test_wheel_install_has_no_deadline_and_is_version_checked(self):
        calls = []

        def fake_group(args, env, *, label="", capture=False):
            calls.append((args, label, capture, env.get("PYTHONPATH", "")))
            if capture:
                return json.dumps({"pydicom": "3.0.1", "timm": "1.0.25",
                                   "pydicom_module": "3.0.1", "timm_module": "1.0.24"}) + "\n"
            return None

        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name in s6.WHEEL_HASHES:
                (Path(d) / name).write_bytes(name.encode())
                paths.append(Path(d) / name)
            hashes = {p.name: s6.sha256(p) for p in paths}
            with patch.object(s6, "WHEEL_HASHES", hashes), patch.object(s6, "_run_group", fake_group):
                with self.assertRaisesRegex(ValueError, "installed dependency versions differ"):
                    s6.install_wheels(paths, Path(d) / "code")
        pip, check = calls
        self.assertIn("pip", pip[0])
        self.assertIn("--no-index", pip[0])
        self.assertIn("--no-deps", pip[0])
        self.assertEqual(pip[1], "S6 wheel install")
        self.assertEqual((check[1], check[2]), ("S6 wheel version check", True))
        self.assertTrue(check[3].startswith(str(Path(d) / "code")))

    def test_run_group_captures_success_and_propagates_failure(self):
        self.assertEqual(s6._run_group([sys.executable, '-c', 'print(42)'], None, capture=True).strip(), '42')
        with self.assertRaisesRegex(RuntimeError, 'exit code 7'):
            s6._run_group([sys.executable, '-c', 'raise SystemExit(7)'], None)

    def test_run_group_posix_new_session_and_group_kill(self):
        _FakePopen.instances.clear()
        killed = []
        with patch.object(s6.os, "name", "posix"), patch.object(s6.subprocess, "Popen", _FakePopen), \
                patch.object(s6.os, "killpg", lambda pid, sig: killed.append((pid, sig)), create=True), \
                patch.object(s6.signal, "SIGKILL", 9, create=True):
            with self.assertRaisesRegex(KeyboardInterrupt, "external cancellation"):
                s6._run_controller(["controller"], {})
        self.assertEqual(_FakePopen.instances[0].kwargs.get("start_new_session"), True)
        self.assertEqual(killed, [(4242, 9)])

    def test_generated_exp189_specs_owner_t4_image_and_three_study_timing(self):
        from tools import exp189_make_kernels as generator

        root = Path(__file__).resolve().parents[1]
        release = json.loads((root / "artifacts/releases/EXP169_A172_v2_0_2_stage/kernel/kernel-metadata.json")
                             .read_text(encoding="utf-8"))
        # Compile local specs only; never execute notebook cells or touch an
        # existing experiment/release stage. Read current generator output.
        with tempfile.TemporaryDirectory(prefix="exp189-test-") as temp:
            output = Path(temp)
            with patch.object(generator, "OUT", output):
                generator.main()
            for name in ("parity", "gold58", "timing"):
                folder = output / name
                meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(meta["id"], f"taeyangg4/exp189-s6-{name}")
                self.assertEqual(meta["title"], f"exp189-s6-{name}")
                self.assertEqual((meta["machine_shape"], meta["docker_image"], meta["enable_internet"]),
                                 (release["machine_shape"], release["docker_image"], False))
            nb = json.loads((output / "timing/exp189_timing.ipynb").read_text(encoding="utf-8"))
        timing = "".join(nb["cells"][2]["source"])
        self.assertNotIn(">= 24", timing)
        self.assertIn("service_components", timing)
        self.assertIn("anatomy_fallbacks", timing)
        lines = timing.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("n_sample = "))
        snippet = "\n".join(lines[start:start + 3])
        for count, expected in ((3, 3), (1, 1), (30, 24)):
            scope = {"costed": list(range(count))}
            exec(snippet, scope)
            self.assertEqual(len(scope["sample"]), expected)
            self.assertEqual(len(set(scope["sample"])), expected)
            self.assertEqual((scope["sample"][0], scope["sample"][-1]), (0, count - 1))
        setup = "".join(nb["cells"][1]["source"])
        self.assertLess(setup.index("observe_code("), setup.index("prepare_code("))

    def test_committed_code_tree_pin_equals_the_observed_kernel_output(self):
        if s6.CODE_TREE_SHA256 is None:
            self.skipTest("source is in the unpinned state; nothing committed to compare")
        observed_path = (Path(__file__).resolve().parents[1] /
                         "artifacts/experiments/EXP189_S6_PORT/observe-output/exp189_code_observed.json")
        self.assertTrue(observed_path.is_file(), "a committed pin needs its observation file")
        record = json.loads(observed_path.read_text(encoding="utf-8"))
        self.assertEqual(record["schema"], "exp189-code-observation-v1")
        self.assertEqual((s6.CODE_TREE_SHA256, s6.CODE_TREE_FILES), (record["tree_sha256"], record["files"]))
        self.assertEqual(s6.tree_digest(record["file_sha256"]), s6.CODE_TREE_SHA256)
        self.assertEqual(len(record["file_sha256"]), s6.CODE_TREE_FILES)
        # The same validation pin-code applied before writing (module pins, anatomy, wheels, closure).
        self.assertEqual(s6.check_observation(record)["tree_sha256"], s6.CODE_TREE_SHA256)
        self.assertTrue(record["wheel_pins_match"] and record["public_module_pins_match"]
                        and record["direct_modules_present"])
        self.assertEqual(record["wheel_sha256"], s6.WHEEL_HASHES)

    def test_service_components_include_anchor_and_fallbacks_are_counted(self):
        controller = {"scored": 4, "workers": [
            {"status": "PASS", "stats": {"decode_s": 2.0, "render_s": 1.0, "anchor_s": 4.0, "gpu_s": 1.0, "rows": 3},
             "anchor_methods": {"skm_tea_joint_avg2": 2, "slab_prior": 1}},
            {"status": "PASS", "stats": {"decode_s": 1.0, "render_s": 1.0, "anchor_s": 2.0, "gpu_s": 0.0, "rows": 1},
             "anchor_methods": {"null": 1}}]}
        service = s6.service_components(controller)
        self.assertEqual(service["per_study"], {"decode_s": .75, "render_s": .5, "anchor_s": 1.5, "gpu_s": .25})
        self.assertEqual(service["service_seconds_per_study"], 3.0)
        anchors = s6.anatomy_fallbacks(controller)
        self.assertEqual((anchors["scored"], anchors["model_anchor"], anchors["fallback"]), (4, 2, 2))
        with self.assertRaisesRegex(ValueError, "worker receipts incomplete"):
            s6.service_components({"scored": 4, "workers": [{"status": "NO_RECEIPT"}]})
        with self.assertRaisesRegex(ValueError, "rows differ"):
            s6.service_components(dict(controller, scored=5))


if __name__ == "__main__":
    unittest.main()
