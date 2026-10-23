"""EXP182 v2.0.0: seed-43 science and authenticated historical records.

Changelog: retired legacy producers; explicit control-full-s43 lifecycle forwarding;
historical source archives are verified as data and never executed.

Sole change against EXP143 v2 S42v2: the seed (42 -> 43), i.e. head init,
window/intensity draws, shuffle and epoch seeds. Same 4,347 studies, L2 labels,
recipe, 16 epochs, fixed SWA [14,11,12] + BN replay, BF16, checkpointing off,
local 4070. New work uses the explicit lifecycle. Historical reads never qualify
a new launch. GPU work remains controlled by the lifecycle and B.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "training/src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from tools import exp143_v2 as base
from tools import exp181_coatnet_f0_bridge as bridge

VERSION, HISTORICAL_VERSION, SCHEMA = "2.0.0", "1.0.0", "exp182-own0949-s43-v1"
EXP = ROOT / "artifacts/experiments/EXP182_OWN0949_S43"
PROTOCOL = EXP / "protocol-v1/manifest.json"
PROTOCOL_DOC = EXP / "PROTOCOL.md"
VALIDATION = PROTOCOL.with_name("validation.json")
QUEUE = PROTOCOL.with_name("queue-exp182-s43.json")
SEED, PRECISION, EPOCHS, UPDATES = 43, "bf16", 16, 4352
BASE_SHA, BASE_PROTOCOL, BASE_PROVENANCE = bridge.BASE_SHA, bridge.BASE_PROTOCOL, bridge.BASE_PROVENANCE
INITIAL_STATE_SHA_S42 = bridge.INITIAL_STATE_SHA
# CPU-computed 2026-10-10 with the frozen build_model/load_imagenet after _cpu_seed(43);
# the same rule reproduces S42v2's 8064b3a5... for seed 42. validate recomputes it.
INITIAL_STATE_SHA = "1a6c176c585f601375c5dc0006d958b2d27388c915c98eb08049e2a283990eb3"
RUN_DIRS = {"local": "artifacts/experiments/EXP182_OWN0949_S43/S43"}
TAG = "EXP182_OWN0949_S43"
DEFAULT_AFTER = bridge.oof_tag("AUG1")
INIT, CORPUS = bridge.INIT, bridge.CORPUS
CODE = list(dict.fromkeys([*bridge.CODE, "tools/exp182_own0949_s43.py",
                           "artifacts/experiments/EXP182_OWN0949_S43/PROTOCOL.md"]))
TARGETS = base.TARGETS
sha256, json_hash, read_json = base.sha256, base.json_hash, base.read_json
save_torch_new, save_torch_atomic = base.save_torch_new, base.save_torch_atomic
IDENTITY_FIELDS = {"schema", "created", "run_dirs", "source_bindings", "content_sha256", "recipe"}
EXTRA_FIELDS = {"version", "seed", "base_protocol_sha256", "base_provenance_sha256",
                "expected_initial_state_sha256"}
ARCHIVE = ROOT / ".local/backups/refactor-step8-s43-5f29395300114990b1724a66c5375e70/index.json"
ARCHIVE_SHA256 = "8110bd14ffff8e1a6f78e07652e3b1629eddd5652b0b0c8ac593efd02cba04fd"


def scoped_path(path):
    return bridge.scoped_path(path, EXP)


def write_new_json(path, value):
    path = scoped_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_bindings():
    return {name: sha256(ROOT / name) for name in CODE}


def registration(baseline, created=None):
    p = copy.deepcopy(baseline)
    p.pop("content_sha256")
    p.update(schema=SCHEMA, version=HISTORICAL_VERSION, created=created or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             recipe=dict(baseline["recipe"], seed_local=SEED), seed=SEED, run_dirs=dict(RUN_DIRS),
             source_bindings=source_bindings(), base_protocol_sha256=BASE_SHA,
             base_provenance_sha256=sha256(BASE_PROVENANCE), expected_initial_state_sha256=INITIAL_STATE_SHA)
    p["content_sha256"] = json_hash(p)
    return p


def assert_seed_only_diff(p, baseline):
    if set(p) - set(baseline) != EXTRA_FIELDS or set(baseline) - set(p):
        raise ValueError("unexpected registration fields")
    changed = {k for k in baseline["recipe"] if p["recipe"].get(k) != baseline["recipe"][k]}
    if set(p["recipe"]) != set(baseline["recipe"]) or changed != {"seed_local"} or p["recipe"]["seed_local"] != SEED:
        raise ValueError("recipe changed beyond the seed")
    for key in set(baseline) - IDENTITY_FIELDS:
        if p[key] != baseline[key]:
            raise ValueError("seed-only control contract changed: " + key)
    if (p["seed"] != SEED or p["run_dirs"] != RUN_DIRS or p["base_protocol_sha256"] != BASE_SHA
            or p["expected_initial_state_sha256"] != INITIAL_STATE_SHA):
        raise ValueError("seed/run/initial-state registration changed")


def execution_settings(p, world, precision, seed, checkpointing=None):
    if (world != 1 or precision != PRECISION or seed != SEED
            or (checkpointing is not None and checkpointing is not False)):
        raise ValueError("EXP182 requires local BF16 seed43, checkpointing OFF")
    settings = base.execution_settings(p, world, precision, seed, False)
    if settings["total_steps"] != UPDATES:
        raise ValueError("EXP182 update schedule differs from S42v2")
    return settings


def registered_run(p, lane="local"):
    if lane != "local" or p.get("run_dirs") != RUN_DIRS:
        raise ValueError("EXP182 permits only its fresh local seed43 run")
    return scoped_path(ROOT / RUN_DIRS["local"])




def load_registration(protocol=PROTOCOL, corpus=None, init_weights=None):
    index = historical_index()
    if Path(protocol).resolve() != (ROOT / index["protocol"]).resolve():
        raise ValueError("EXP182 protocol path changed")
    check_historical_file(protocol, index)
    p = read_json(protocol)
    digest = p.pop("content_sha256", None)
    if json_hash(p) != digest or p.get("schema") != SCHEMA or p.get("version") != HISTORICAL_VERSION:
        raise ValueError("EXP182 registration content changed")
    p["content_sha256"] = digest
    baseline = bridge.baseline_registration(corpus, init_weights)
    assert_seed_only_diff(p, baseline)
    if (p["source_bindings"] != {s["path"]: s["sha256"] for s in index["sources"]}
            or p["base_provenance_sha256"] != sha256(BASE_PROVENANCE)):
        raise ValueError("EXP182 source/input binding changed")
    registered_run(p)
    return p


def require_validation(protocol=PROTOCOL):
    load_registration(protocol)
    check_historical_file(VALIDATION, historical_index())
    v = read_json(VALIDATION)
    if (v.get("status") != "VALIDATED_NO_TRAINING" or v.get("protocol_sha256") != sha256(protocol)
            or v.get("initial_state_sha256") != INITIAL_STATE_SHA
            or not all(v.get(k) for k in ("actual_pin_memory_tuple_returns_list", "full_corpus_hashes",
                "real_backbone_keys_loaded_cpu", "seed_only_diff_verified"))):
        raise ValueError("EXP182 CPU validation receipt missing/stale")
    return v










def check_summary(summary, p, protocol_sha, model_sha):
    identity = summary.get("identity", {})
    if (summary.get("status") != "COMPLETE_TRAINING_NO_EVALUATION"
            or summary.get("epochs") != EPOCHS or summary.get("updates") != UPDATES
            or summary.get("skips") != 0 or [h.get("epoch") for h in summary.get("history", [])] != list(range(1, 17))
            or summary.get("model_sha256") != model_sha or identity.get("protocol_sha256") != protocol_sha
            or identity.get("initial_state_sha256") != INITIAL_STATE_SHA or identity.get("seed") != SEED
            or identity.get("run_dir") != RUN_DIRS["local"]
            or identity.get("execution") != execution_settings(p, 1, PRECISION, SEED, False)):
        raise ValueError("EXP182 fit incomplete or identity changed")
    return summary


def require_completed(protocol=PROTOCOL):
    p = load_registration(protocol)
    require_validation(protocol)
    run = registered_run(p)
    index = historical_index()
    for name in ("summary.json", "provenance.json", "model_best.pt"):
        check_historical_file(run / name, index)
    summary = check_summary(read_json(run / "summary.json"), p, sha256(protocol), sha256(run / "model_best.pt"))
    provenance = read_json(run / "provenance.json")
    if any(provenance.get(key) != value for key, value in summary["identity"].items()):
        raise ValueError("EXP182 provenance identity changed")
    return summary


def historical_index():
    """Authenticate original sources as inert bytes, independent of live wrappers."""
    if sha256(ARCHIVE) != ARCHIVE_SHA256:
        raise ValueError("EXP182 historical archive index changed")
    index = read_json(ARCHIVE)
    if (index.get("schema") != "exp182-historical-source-archive-v1"
            or index.get("version") != "1.0.0" or index.get("historical_only") is not True):
        raise ValueError("EXP182 historical archive identity changed")
    for spec in index["sources"]:
        path = (ARCHIVE.parent / spec["archive"]).resolve()
        if (ARCHIVE.parent.resolve() not in path.parents or path.stat().st_size != spec["bytes"]
                or sha256(path) != spec["sha256"]):
            raise ValueError("EXP182 historical source archive changed")
    return index


def check_historical_file(path, index):
    path = Path(path).resolve()
    try:
        name = str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError as exc:
        raise ValueError("EXP182 historical artifact path changed") from exc
    pin = index["artifacts"].get(name)
    if not pin or path.stat().st_size != pin["bytes"] or sha256(path) != pin["sha256"]:
        raise ValueError("EXP182 historical artifact changed: " + name)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "lifecycle":
        from tools.coatnet_config import main as lifecycle_main
        return lifecycle_main(argv[1:], preset="control-full-s43")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("status",))
    ap.add_argument("--protocol", default=str(PROTOCOL))
    if argv and argv[0] in {"register", "prepare", "validate", "queue-json", "train"}:
        ap.error("legacy producer retired; use lifecycle <command> with a fresh control-full-s43 stage")
    args = ap.parse_args(argv)
    result = dict(status=require_completed(args.protocol)["status"],
                  historical_only=True, launch_authorized=False)
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    result = main()
    sys.exit(result if isinstance(result, int) else 0)




