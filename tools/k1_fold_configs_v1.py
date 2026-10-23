"""H224 fold-ensemble config materializer v1.0.0; local files/CPU only.

Changelog: five/six-model templates, verified asset pins and <=8h projection.
Templates are deliberately not executable K1N configs. No fabricated hashes.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from tools import k1_fold_dataset_v1 as assets
from tools import k1n_config_v1 as schema

VERSION = "1.0.0"
ROOT = assets.ROOT
BASE_DIRS = (
    ROOT / ".local/scratch/slot-b/k2-release/fix1/assets-v1",
    ROOT / ".local/scratch/astra-b1/k1-shared-render/dataset-k1-v3gs0",
    ROOT / ".local/scratch/astra-b1/k1-shared-render/dataset-k1-v3gs1",
)
need = assets.need


def template(path):
    value = assets.read(path)
    need(set(value) == {"schema", "model_count", "folds", "drop_o", "alpha", "k1_eval_stride",
                        "encoder_chunk", "runtime"}, "template fields")
    need(value["schema"] == "k1-fold-template/v1", "template schema")
    n = value["model_count"]
    need(type(n) is int and n in (5, 6) and value["folds"] == list(range(1, n - 2)),
         "registered five/six-model fold order")
    need(value["drop_o"] is True and value["alpha"] == .25
         and type(value["k1_eval_stride"]) is int and value["k1_eval_stride"] == 2
         and type(value["encoder_chunk"]) is int and value["encoder_chunk"] == 32,
         "fixed drop-O/alpha/stride/chunk contract")
    runtime = value["runtime"]
    need(set(runtime) == {"parent_hours", "parent_k1_models_stride3", "per_model_stride3_hours",
                          "stride2_multiplier", "o_hours", "limit_hours", "basis"},
         "runtime fields")
    need(runtime["parent_hours"] == [6.5, 6.9] and runtime["per_model_stride3_hours"] == .45
         and runtime["stride2_multiplier"] == 1.44 and runtime["o_hours"] == 1.1
         and runtime["limit_hours"] == 8.0, "registered runtime assumptions changed; version the tool")
    need(type(runtime["parent_k1_models_stride3"]) is int
         and runtime["parent_k1_models_stride3"] in (0, 3), "parent scope must be explicit")
    need(isinstance(runtime["basis"], str) and len(runtime["basis"].strip()) >= 20,
         "runtime assumption evidence required")
    return value


def projection(value):
    r = value["runtime"]
    per_model = Decimal(str(r["per_model_stride3_hours"]))
    removed = r["parent_k1_models_stride3"] * per_model
    added = value["model_count"] * per_model * Decimal(str(r["stride2_multiplier"]))
    exact = [Decimal(str(h)) - Decimal(str(r["o_hours"])) - removed + added for h in r["parent_hours"]]
    interval = [float(h) for h in exact]
    return dict(schema="k1-fold-runtime/v1", version=VERSION,
                status="PASS_ESTIMATE_ONLY" if exact[1] <= Decimal(str(r["limit_hours"])) else "HOLD",
                model_count=value["model_count"], projected_hours=interval, limit_hours=r["limit_hours"],
                equation="parent_hours - O_hours - parent_K1_count*0.45 + model_count*0.45*1.44",
                removed_parent_k1_hours=float(removed), added_k1_hours=float(added),
                per_model_stride2_hours=.648, assumptions=r,
                measured_hidden_runtime=False, qualified_for_submission=False,
                note="Upper endpoint controls admission. Estimates exclude unmeasured overhead; "
                     "gross O saving and linear classifier scaling still need real qualification.")


def materialize(value, folds, kernel_id, release_version, base_dirs=BASE_DIRS):
    runtime = projection(value)
    need(runtime["status"] != "HOLD",
         f"runtime upper projection {runtime['projected_hours'][1]:.3f}h exceeds 8.0h; no config emitted")
    need(len(base_dirs) == 3 and len(folds) == len(value["folds"]), "exact asset directory count")
    models = schema.current()["models"]
    for directory, model in zip(base_dirs, models):
        for name, pin in (("asset-manifest.json", model["manifest_sha256"]),
                          (model["checkpoint"], model["checkpoint_sha256"]),
                          (model["trainer_source"], model["trainer_sha256"])):
            need(assets.sha(assets.member(directory, name)) == pin, "base asset identity mismatch")
    for fold, directory in zip(value["folds"], folds):
        model = assets.verify_dataset(directory, f"k1_v3g_fold{fold}")
        manifest = assets.read(Path(directory) / "asset-manifest.json")
        need(manifest.get("schema") == "k1-v3g-fold-assets/v1"
             and manifest.get("model_fold") == fold and manifest.get("model_seed") == 0,
             "fold dataset identity/order mismatch")
        need(schema.valid_hash(manifest.get("training_contract"))
             and schema.valid_hash(manifest.get("export_sha256"))
             and schema.valid_hash(manifest.get("training_source_sha256")), "fold export provenance missing")
        models.append(model)
    config = schema.validate(dict(schema="k1n-config/v1", models=models,
                                  kernel_id=kernel_id, release_version=release_version,
                                  drop_o=True, alpha=.25, k1_eval_stride=2, encoder_chunk=32))
    runtime.update(config_sha256=hashlib.sha256(assets.encoded(config)).hexdigest(),
                   template_sha256=hashlib.sha256(assets.encoded(value)).hexdigest())
    return config, runtime


def run(action, template_path, folds, output, kernel_id, release_version, base_dirs=BASE_DIRS):
    value = template(template_path)
    config, runtime = materialize(value, folds, kernel_id, release_version, base_dirs)
    output = Path(output).absolute()
    assets.member(output, "config.json")
    payloads = {"config.json": assets.encoded(config), "runtime-projection.json": assets.encoded(runtime)}
    if action == "prepare":
        need(not output.exists(), "output exists; use validate or a new version")
        output.mkdir(parents=True)
        for name, data in payloads.items():
            (output / name).write_bytes(data)
    else:
        need(action == "validate", "unknown action")
    need({p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == set(payloads),
         "config directory inventory drift")
    for name, data in payloads.items():
        need(assets.member(output, name).read_bytes() == data, "config/runtime artifact drift")
    return dict(status="PASS_LOCAL_CONFIG", config=str(output / "config.json"),
                runtime=runtime, qualified_for_submission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("project", "prepare", "validate"))
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--fold-datasets", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--kernel-id")
    parser.add_argument("--release-version")
    args = parser.parse_args()
    try:
        value = template(args.template)
        if args.action == "project":
            result = projection(value)
            print(json.dumps(result, indent=2))
            if result["status"] == "HOLD":
                parser.exit(2)
            return
        need(args.fold_datasets and args.output and args.kernel_id and args.release_version,
             "prepare/validate require --fold-datasets, --output, --kernel-id, --release-version")
        print(json.dumps(run(args.action, args.template, args.fold_datasets, args.output,
                             args.kernel_id, args.release_version), indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, "HOLD: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
