"""v1.0.0 HOST_BOUND_EXPORT for next all-data jobs; exact foldgate tolerances."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
HOST_FLOAT_FIELDS = ("constant_loss", "threshold")
HOST_PATH_FLAGS = ("--px", "--targets", "--out", "--init")
HOST_FLOAT_ATOL = 1e-12


def prepared_hashes(reg, backend=None):
    """Reproduce prepare() bytes in memory; never write training artifacts."""
    k = backend or globals()["k"]
    rows = k.training_rows(reg)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    source = (k.ROOT / k.s.SOURCE / "compact_train.py").read_text(encoding="utf-8")
    return dict(targets_sha256=hashlib.sha256(stream.getvalue().encode("utf-8")).hexdigest(),
                trainer_sha256=hashlib.sha256(k.derive_trainer(source).encode("utf-8")).hexdigest())


def validate_host_export(paths, result, expected, fold, backend=None):
    """Only explicitly enumerated host fields may differ; all else is exact."""
    k = backend or globals()["k"]
    base = Path(paths["k1_result"]).resolve().parent.parent
    export_path = base / "EXPORT.json"
    export = k.read(export_path)
    k.need(export["fold"] == fold and export["contract_id"] == result["contract_id"],
           "host export fold/contract mismatch")
    bound = {"registration.json": base / "registration.json",
             "prepare.json": base / "prepare.json",
             "fit/result.json": Path(paths["k1_result"]),
             "fit/train-guard.csv": Path(paths["k1_csv"])}
    for name, path in bound.items():
        k.need(path.resolve() == (base / name).resolve()
               and export["sha256"][name] == k.sha(path), "host export hash/path mismatch")
    reg = k.read(bound["registration.json"])
    k.need(reg["contract_id"] == result["contract_id"]
           and k.digest({n: v for n, v in reg.items() if n != "contract_id"}) == reg["contract_id"],
           "host registration digest/contract mismatch")
    normalized = json.loads(json.dumps(reg))
    differences = []
    for field in HOST_FLOAT_FIELDS:
        actual, local = reg["collapse_guard"][field], expected["collapse_guard"][field]
        k.need(type(actual) in (int, float) and math.isfinite(actual)
               and abs(actual - local) <= HOST_FLOAT_ATOL, "host collapse constant mismatch")
        if actual != local:
            differences.append("registration.collapse_guard." + field)
        normalized["collapse_guard"][field] = local
    normalized["contract_id"] = expected["contract_id"]
    # Includes exact ordered UID lists, all pins, recipe and every unknown field.
    k.need(normalized == expected, "host registration scientific identity mismatch")
    prep = k.read(bound["prepare.json"])
    k.need(set(prep) == {"contract_id", "trainer_sha256", "targets_sha256",
                        "train_count", "holdout_training_rows", "argv", "cache"}
           and prep["contract_id"] == reg["contract_id"]
           and prep["train_count"] == len(expected["train"])
           and prep["holdout_training_rows"] == 0, "host prepare identity mismatch")
    hashes = prepared_hashes(expected) if backend is None else prepared_hashes(expected, backend)
    k.need(all(prep[n] == value for n, value in hashes.items()),
           "host prepared targets/trainer hash mismatch")
    local_argv = k.trainer_argv(k.CACHE, expected)
    argv = prep["argv"]
    k.need(isinstance(argv, list) and len(argv) == len(local_argv), "host argv shape mismatch")
    path_fields = {0: "python", 1: "trainer",
                   **{local_argv.index(flag) + 1: flag for flag in HOST_PATH_FLAGS}}
    guard_index = local_argv.index("--l1-guard-constant") + 1
    for index, local in enumerate(local_argv):
        actual = argv[index]
        if index in path_fields:
            k.need(isinstance(actual, str) and
                   (PurePosixPath(actual).is_absolute() or PureWindowsPath(actual).is_absolute()),
                   "host argv path must be absolute")
            if actual != local:
                differences.append("prepare.argv." + path_fields[index])
        elif index == guard_index:
            k.need(actual == repr(reg["collapse_guard"]["constant_loss"]),
                   "host argv collapse constant mismatch")
            if actual != local:
                differences.append("prepare.argv.--l1-guard-constant")
        else:
            k.need(actual == local, "host argv recipe mismatch")
    k.need(prep["cache"] == argv[local_argv.index("--px") + 1], "host cache/argv mismatch")
    if prep["cache"] != str(Path(k.CACHE).resolve()):
        differences.append("prepare.cache")
    return dict(mode="HOST_BOUND_EXPORT", contract_id=reg["contract_id"],
                local_expected_contract_id=expected["contract_id"],
                differing_fields=differences, collapse_constant_atol=HOST_FLOAT_ATOL,
                export_sha256=k.sha(export_path),
                files={n: export["sha256"][n] for n in bound})

