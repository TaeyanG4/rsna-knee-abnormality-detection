"""Local CPU engineering smoke; never changes the qualified CUDA/fp16 runtime."""
from __future__ import annotations

import csv
import gc
import hashlib
import os
from pathlib import Path
import time

try:
    from tools import exp169_release_p956 as parent, exp169_release_p956rplus as release
    from tools import exp152_own0949_release as helper
except ImportError:
    import exp169_release_p956 as parent
    import exp169_release_p956rplus as release
    import exp152_own0949_release as helper

SCRATCH = release.ROOT / ".local/scratch/sol-1/p956-rplus"
BASE_WEIGHTS = release.ROOT / ".local/rplus-v1/public-R-download.bin"
COMP = release.ROOT / "storage/competition"
DATA_ZIP = release.ROOT / "storage/downloads/rsna-knee-abnormality-detection/rsna-knee-abnormality-detection.zip"
IMAGES = SCRATCH / "dicom/test_series"


def cpu_reader(fastread):
    """Windows transport only: each fastread job exclusively owns its file descriptor."""
    if not hasattr(os, "pread"):
        fastread._O_FLAGS |= getattr(os, "O_BINARY", 0)
        def pread(fd, size, offset):
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, size)
        fastread._pread = pread
    return fastread


def replay(output: Path):
    import numpy as np
    import pandas as pd
    import torch

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    release.need(not (output / "replay.json").exists(), "stale CPU replay")
    release.public_spec()
    release.source_files()
    parent.check_pin(BASE_WEIGHTS, parent.R_SHA)
    release.check_pin(release.R_SOURCE, release.R_SHA)
    with (COMP / "test.csv").open(encoding="utf-8-sig", newline="") as stream:
        ids = [row[release.UID] for row in csv.DictReader(stream)][:2]
    release.need(len(set(ids)) == 2, "two local studies required")
    import zipfile
    with zipfile.ZipFile(DATA_ZIP) as archive:
        for name in archive.namelist():
            if name.startswith("test_series/") and name.endswith(".dcm") and any("/" + uid + "/" in name for uid in ids):
                release.need(".." not in Path(name).parts, "invalid DICOM archive path")
                archive.extract(name, SCRATCH / "dicom")
    cols = [release.UID, "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"]
    frame = pd.read_csv(COMP / "test_series.csv", usecols=cols,
                        dtype={release.UID: str, "SeriesInstanceUID": str})
    records = {uid: group.to_dict("records") for uid, group in frame.groupby(release.UID, sort=False)}
    ns = {"__name__": "p956rplus_cpu_serving", "__file__": str(SCRATCH / "public-serving.py")}
    source = (release.VISIBLE / "exp155-assets/public-serving.py").read_text(encoding="utf-8")
    exec(compile(release.r_runtime_source(source), ns["__file__"], "exec"), ns)

    # CPU-only accommodation: CUDA autocast cannot run here. All model, window,
    # K94, mirror and pair-averaging code stays the actual serving implementation.
    @torch.inference_mode()
    def cpu_probs(model, windows, device):
        assert device == "cpu"
        logits = model(windows.unsqueeze(0)).float()
        release.need(bool(torch.isfinite(logits).all()), "nonfinite CPU logits")
        return torch.sigmoid(logits)[0].numpy()

    ns["infer_probs"] = cpu_probs
    reader = cpu_reader(ns["_load_fastread"]()).make_builder(ns["FASTREAD_VARIANT"], 384, ns["SLOTS"],
                                               hdr_threads=4, px_threads=4)
    inputs = {}
    for index, uid in enumerate(ids):
        vol, mask = reader(uid, records, str(IMAGES))
        release.need(bool(mask.any()), "empty local decoded study")
        np.savez(SCRATCH / f"study-{index}.npz", volume=vol, mask=mask)
        inputs[uid] = dict(volume_sha256=helper.array_sha(vol), mask_sha256=helper.array_sha(mask))
        print(f"decoded local study {index + 1}/2", flush=True)
    results = {}
    schemas = {}
    for label, module, weights in (("R", parent, BASE_WEIGHTS), ("Rplus", release, release.R_SOURCE)):
        module.bind_r(ns, helper, weights, COMP, SCRATCH)
        model, res = ns["_E152"].checked_model(ns, weights, helper.ARCH, 384, "cpu")
        release.need(res == ns["INPUT_RES"] == 320 and ns["K_EVAL"] == 94, "serving contract drift")
        schemas[label] = {k: (list(v.shape), str(v.dtype)) for k, v in model.state_dict().items()}
        predictions = []
        print(f"{label}: strict-loaded serving model at 320", flush=True)
        for index, uid in enumerate(ids):
            with np.load(SCRATCH / f"study-{index}.npz", allow_pickle=False) as data:
                vol, mask = data["volume"], data["mask"]
            plain = ns["gpu_windows"](vol, mask, 94, 320, "cpu")
            mirror = ns["gpu_windows"](vol, mask, 94, 320, "cpu", mirror=True)
            hashes = dict(plain_sha256=helper.array_sha(plain), mirror_sha256=helper.array_sha(mirror))
            if label == "R":
                inputs[uid].update(hashes)
            else:
                release.need(all(inputs[uid][k] == v for k, v in hashes.items()), "R/R+ windows differ")
            values = ns["infer_pair_probs"](model, plain, mirror, "cpu")
            predictions.append(release.RBinding.prediction(values))
            del plain, mirror
            print(f"{label}: CPU plain+mirror study {index + 1}/2 finite", flush=True)
        results[label] = np.asarray(predictions)
        del model
        gc.collect()
    release.need(schemas["R"] == schemas["Rplus"], "R/R+ serving state schemas differ")

    # The S6 path is unchanged: authenticate and replay its real Kaggle predictions.
    prior = parent.read_json(parent.REPLAY / "p956-receipt.json")
    s6_path = parent.REPLAY / "p956-run/s6/predictions.csv"
    parent.check_pin(s6_path, prior["provenance"]["s6_raw_sha256"])
    all_ids, all_s6 = parent.v5q._read_csv(s6_path)
    ours = all_s6[[all_ids.index(uid) for uid in ids]].copy()
    for name, values in (("baseline-r.csv", results["R"]), ("rplus.csv", results["Rplus"]),
                         ("baseline-s6.csv", ours), ("rplus-s6.csv", ours.copy())):
        release.write_once(output / name, release.csv_bytes(ids, values))
    before = parent.blend(ids, results["R"], ours)
    after = release.blend(ids, results["Rplus"], ours)
    expected = parent.public_formula(results["Rplus"], ours)
    release.need(np.array_equal(after, expected), "public formula drift")
    changed = np.any(results["R"] != results["Rplus"], axis=1)
    release.need(bool(changed.all()), "R predictions must change in each study")
    release.write_once(output / "baseline-submission.csv", parent.csv_bytes(ids, before))
    release.write_once(output / "submission.csv", release.csv_bytes(ids, after))
    release.write_once(output / "inputs.json", release.json_bytes(inputs))
    report = dict(status="PASS_CPU_REPLAY", studies=2, findings=12, strict_load=True,
        state_tensors=len(schemas["Rplus"]), native_resolution=320, k_eval=94, tta=["plain", "mirror"],
        device="cpu", precision="float32", finite=True, r_changed_each_study=bool(changed.all()),
        changed_r_values=int(np.count_nonzero(results["R"] != results["Rplus"])),
        r_max_abs=float(np.max(np.abs(results["R"] - results["Rplus"]))),
        s6_exact=(output / "baseline-s6.csv").read_bytes() == (output / "rplus-s6.csv").read_bytes(),
        public_formula_exact=True, r_sha256=parent.R_SHA, rplus_sha256=release.R_SHA,
        s6_source_sha256=parent.sha(s6_path), test_csv_sha256=parent.sha(COMP / "test.csv"),
        series_csv_sha256=parent.sha(COMP / "test_series.csv"),
        serving_sha256=release.PINS["public-serving.py"],
        submission_sha256=parent.sha(output / "submission.csv"),
        files={p.name: parent.sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        seconds=round(time.monotonic()-start, 3),
        scope="Fresh R/R+ inference on two local DICOM studies in CPU float32 with Windows binary positioned-read shim; authenticated S6 replay unchanged. "
              "No fresh S6 inference; not CUDA/fp16 parity, accuracy evaluation or hidden-runtime qualification.")
    release.write_once(output / "replay.json", release.json_bytes(report))
    return report
