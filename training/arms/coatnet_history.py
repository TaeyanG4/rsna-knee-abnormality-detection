"""v1.0.0: authenticated, read-only access to retired CoAtNet records.

Changelog: immutable source archives replace mutable live-wrapper comparisons.
These records are historical evidence, never lifecycle or launch qualifications.
Archived Python is hashed as data and is never imported or executed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = "1.0.0"
ARCHIVE = ".local/backups/refactor-step8-history-0f8ad6311feb4bbd86ffc0193b0d5042/index.json"
ARCHIVE_SHA256 = "043c5475cbd64fe83fcbf956bde99af38dc950921cd9a200b1695fe1b370ff03"
INITIAL_STATE_SHA256 = "8064b3a56c602e5e52abd3a10b77f235300dd7008f56fc4d7d47fca75e5d3c43"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OOF_PLAIN, OOF_PRIMARY = "oof_k94.csv", "oof_k94_mirror_tta.csv"
OOF_RECEIPT, INVENTORY = "oof-receipt.json", "output-inventory.json"
CONTROL = "artifacts/experiments/EXP181_COATNET_F0_BRIDGE/protocol-v1/manifest-CONTROL.json"
# Registration pins live in the independently pinned archive index. Artifact
# pins are separate: a coherent rewrite of a receipt + inventory is not history.
VALIDATIONS = {
    "EXP181_COATNET_F0_BRIDGE": "eebfb83122f10bc61083d3afb2082b166661bd95b325e17e8e0c50f78bf1588c",
    "EXP188_COATNET_COVERAGE_F0": "b99c7cae93a02219ec20745b8e868ed06f355e6d84860ac276f9c683cadbccac",
    "EXP192_COATNET_CROP_F0": "9c0076890c20175a59ef62481e173f4816c83d6b79fa27bafb0ddd831e91d4b6",
    "EXP192B_COATNET_CROP_F0": "02d8269b8d1e19e99163d3189ac36b0aff6350dbf3c7bbfbcff2d7abe866208f",
    "EXP190E_COATNET_OAI_F0": "9d45eed5dea9d4e41187797b4424fe25f5e9bbfe24cf23db3ee74db1ee7e3edd",
}
INVENTORIES = {
    "artifacts/experiments/EXP181_COATNET_F0_BRIDGE/CONTROL": "1178b02d42f684d86f00d1549fed03026ca91bb209686f507ef40b0597dc9338",
    "artifacts/experiments/EXP181_COATNET_F0_BRIDGE/AUG1": "79d63adc02776951fcef7b49a6b6de6d38f47370e6a9d65502bc7178ee1945f0",
    "artifacts/experiments/EXP188_COATNET_COVERAGE_F0/COVERAGE": "e4d71dc0329a1edf59a3b5e3b39797059808b3bf300b254433bc5255ec91fc70",
    "artifacts/experiments/EXP192B_COATNET_CROP_F0/CROP": "7e5b6eac5c40fe496e9e3b56e7d3e88865f4fb8a9f447d1c4ca0b69aed730768",
}
CROP_TIMING = {
    "fit-wall.json": "e1baf33c9e3ee87cee41a455a507252e0f9f10721707eff6cf1f6a86cce8f7ec",
    "total-wall.json": "53b313346cdb7a46720151a1cbc5d9fb6668940ac9f024fa9a4bf01ae635e2f5",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def uid_hash(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _contained(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    _need(root in path.parents, "historical path escapes its root")
    return path


def _index(root):
    path = _contained(root, ARCHIVE)
    _need(sha256(path) == ARCHIVE_SHA256, "historical archive index changed")
    index = read_json(path)
    _need(index.get("schema") == "coatnet-historical-source-archive-v1"
          and index.get("version") == "1.0.0" and index.get("historical_only") is True,
          "historical archive schema differs")
    return path.parent, index


def read_registration(protocol, *, root=ROOT):
    """Read one of six pinned original manifests; authenticate original sources."""
    root = Path(root).resolve()
    protocol = _contained(root, protocol)
    archive, index = _index(root)
    matches = [m for m in index["manifests"] if _contained(root, m["path"]) == protocol]
    _need(len(matches) == 1, "unregistered historical manifest path")
    spec = matches[0]
    _need(sha256(protocol) == spec["sha256"], "historical manifest bytes changed")
    p = read_json(protocol)
    digest = p.pop("content_sha256", None)
    _need(json_hash(p) == digest, "historical manifest content changed")
    p["content_sha256"] = digest
    _need(all(p.get(k) == spec[k] for k in ("schema", "version", "arm", "run_dirs")),
          "historical manifest identity changed")
    blobs = {(s["path"], s["sha256"]): s for s in index["sources"]}
    for field in ("source_bindings", "screen_source_bindings"):
        expected = {s["path"]: s["sha256"] for s in spec["sources"] if s["field"] == field}
        _need(p.get(field) == expected, "historical source map differs")
        for name, digest in expected.items():
            entry = blobs[(name, digest)]
            blob = _contained(archive, entry["archive"])
            _need(blob.stat().st_size == entry["bytes"] and sha256(blob) == digest,
                  "historical source archive blob changed: " + name)
    train, val = p["train_ids"], p["val_ids"]
    _need(len(val) == len(set(val)) == 888 and len(train) == len(set(train))
          and not set(train) & (set(val) | set(p["excluded_gold_ids"]))
          and not set(val) & set(p["excluded_gold_ids"]), "historical cohort differs")
    _need(len(train) == (len(p["rsna_train_ids"]) + len(p["oai_train_ids"])
                        if p["arm"] == "OAI" else 3455), "historical training cohort differs")
    _need(uid_hash(val) == p["fold"]["val_sha256"], "historical validation order differs")
    _contained(root, p["run_dirs"]["local"])
    return p


def require_validation(protocol, *, root=ROOT):
    """Authenticate the original CPU receipt without conferring launch approval."""
    p = read_registration(protocol, root=root)
    protocol = _contained(root, protocol)
    receipt = protocol.parent / "validation.json"
    expected = VALIDATIONS.get(protocol.parent.parent.name)
    _need(expected is not None and sha256(receipt) == expected, "historical CPU validation changed")
    v = read_json(receipt)
    bound = v.get("manifest_sha256")
    if isinstance(bound, dict):
        bound = bound.get(p["arm"])
    _need(v.get("status") == "VALIDATED_NO_TRAINING" and bound == sha256(protocol)
          and v.get("initial_state_sha256") == INITIAL_STATE_SHA256,
          "historical CPU validation identity differs")
    return v


def historical_execution(p):
    """Original accounting is part of provenance, never a cap on new execution."""
    settings = dict(lane="local", batch=1, accumulation=16, effective_batch=16,
                    grad_checkpointing=False, microbatches_per_epoch=3455,
                    updates_per_epoch=216, total_steps=3456)
    if "budgets" in p:
        settings.update(max_hours=p["budgets"]["local"], reserve_seconds=p["budgets"]["reserve_seconds"])
    return settings


def check_summary(summary, p, protocol_sha, model_sha):
    ident = summary.get("identity", {})
    expected = dict(protocol_sha256=protocol_sha, initial_state_sha256=INITIAL_STATE_SHA256,
                    seed=42, precision="bf16", world_size=1, contract="exp143_raptor96_384_c140_s02",
                    train_n=len(p["train_ids"]), torch="2.5.1+cu121", run_dir=p["run_dirs"]["local"],
                    execution=historical_execution(p))
    _need(summary.get("status") == "COMPLETE_TRAINING_NO_EVALUATION"
          and summary.get("epochs") == 16 and summary.get("updates") == 3456
          and summary.get("skips") == 0 and summary.get("model_sha256") == model_sha
          and [h.get("epoch") for h in summary.get("history", [])] == list(range(1, 17))
          and ident == expected, "historical completed summary differs")
    return summary


def check_checkpoint(checkpoint, p, protocol_sha):
    r = p["recipe"]
    _need(checkpoint.get("arch") == r["arch"]
          and checkpoint.get("contract") == "exp143_raptor96_384_c140_s02"
          and checkpoint.get("oai") is (p["arm"] == "OAI")
          and checkpoint.get("swa_epochs_1based") == r["swa_1based"]
          and checkpoint.get("lab") == TARGETS and checkpoint.get("checkpoint_kind") == "fixed-swa-bn"
          and checkpoint.get("seed") == 42
          and checkpoint.get("provenance", {}).get("protocol_sha256") == protocol_sha
          and checkpoint["provenance"].get("run_dir") == p["run_dirs"]["local"],
          "historical checkpoint identity differs")
    if p["arm"] == "CROP":
        _need(checkpoint.get("window_policy") == "exp192-alternating-c140-c100"
              and checkpoint.get("crop_policy") == p["augmentation"], "historical crop checkpoint policy differs")


def load_oof(path, ids):
    """Validate exact cohort/target order and finite probabilities without output."""
    import numpy as np
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _need(reader.fieldnames == ["StudyInstanceUID", *TARGETS], "historical OOF columns differ")
        rows = list(reader)
    by_id = {row["StudyInstanceUID"]: row for row in rows}
    _need(len(by_id) == len(rows) == len(ids) and set(by_id) == set(ids), "historical OOF cohort differs")
    values = np.asarray([[float(by_id[uid][target]) for target in TARGETS] for uid in ids], dtype=np.float64)
    _need(values.shape == (len(ids), 12) and np.isfinite(values).all()
          and ((values >= 0) & (values <= 1)).all(), "historical OOF probabilities invalid")
    return values


def verify_record(p, run, protocol_sha, *, summary_checker, checkpoint_checker):
    """Shared explicit completed-artifact checker; does not read or patch globals.

    The caller authenticates its registration/validation first. Historical callers
    additionally pin the inventory itself. Current engineering callers retain
    their own source-bound registration and explicit scientific checkers.
    """
    run = Path(run).resolve()
    inventory = read_json(run / INVENTORY)
    required = {"summary.json", "provenance.json", "model_best.pt", OOF_PLAIN, OOF_PRIMARY, OOF_RECEIPT}
    _need(required <= set(inventory), "completed inventory missing required member")
    for name, entry in inventory.items():
        path = _contained(run, name)
        _need(path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"],
              "completed inventory member changed: " + name)
    model_sha = inventory["model_best.pt"]["sha256"]
    summary = read_json(run / "summary.json")
    summary_checker(summary, p, protocol_sha, model_sha)
    provenance = read_json(run / "provenance.json")
    _need(all(provenance.get(k) == v for k, v in summary["identity"].items()),
          "completed provenance differs from summary")
    _need(provenance.get("swa_epochs_1based") == p["recipe"]["swa_1based"]
          and provenance.get("oai") is (p["arm"] == "OAI")
          and provenance.get("grad_checkpointing") is False, "completed provenance recipe differs")
    receipt = read_json(run / OOF_RECEIPT)
    expected = dict(status="PREDICTED", arm=p["arm"], fold=0, rows=888,
                    cohort_sha256=uid_hash(p["val_ids"]), protocol_sha256=protocol_sha,
                    bridge_contract_sha256=p["bridge_contract_sha256"], checkpoint_sha256=model_sha,
                    summary_sha256=inventory["summary.json"]["sha256"], initial_state_sha256=INITIAL_STATE_SHA256,
                    primary=OOF_PRIMARY, k_eval=p["recipe"]["eval_k"],
                    inference_res=p["recipe"]["inference_res"], precision="fp16")
    _need(all(receipt.get(k) == v for k, v in expected.items()), "completed OOF receipt differs")
    _need(receipt.get("outputs") == {n: inventory[n]["sha256"] for n in (OOF_PLAIN, OOF_PRIMARY)},
          "completed OOF output bindings differ")
    _need(receipt.get("torch") == summary["identity"].get("torch")
          and isinstance(receipt.get("device_name"), str) and bool(receipt["device_name"]),
          "completed runtime receipt differs")
    _need(isinstance(receipt.get("seconds"), (int, float)) and math.isfinite(receipt["seconds"])
          and receipt["seconds"] > 0, "completed OOF timing invalid")
    for name in (OOF_PLAIN, OOF_PRIMARY):
        load_oof(run / name, p["val_ids"])
    import torch
    checkpoint = torch.load(run / "model_best.pt", map_location="cpu", weights_only=False)
    checkpoint_checker(checkpoint, p, protocol_sha)
    _need(checkpoint.get("provenance") == summary["identity"], "checkpoint provenance differs from completed identity")
    return dict(p=p, run=run, receipt=receipt, protocol_sha256=protocol_sha, launch_authorized=False)


def verified_arm(protocol, *, root=ROOT):
    """Authenticate a completed historical record; incomplete originals fail closed."""
    p = read_registration(protocol, root=root)
    require_validation(protocol, root=root)
    run = _contained(root, p["run_dirs"]["local"])
    pin = INVENTORIES.get(p["run_dirs"]["local"])
    _need(pin is not None, "historical run has no authenticated complete output inventory")
    _need(sha256(run / INVENTORY) == pin, "historical output inventory changed")
    digest = sha256(_contained(root, protocol))
    if p["arm"] == "CROP":
        for name, pin in CROP_TIMING.items():
            _need(sha256(run.parent / name) == pin, "historical crop timing receipt changed")
        fit, wall = (read_json(run.parent / n) for n in ("fit-wall.json", "total-wall.json"))
        values = [fit["seconds"], wall["fit_seconds"], wall["oof_seconds"], wall["total_seconds"]]
        _need(all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values)
              and fit.get("status") == wall.get("status") == "COMPLETE"
              and fit.get("manifest_sha256") == wall.get("manifest_sha256") == digest
              and fit.get("includes_pilot") is True and fit["seconds"] == wall["fit_seconds"]
              and wall["total_seconds"] == wall["fit_seconds"] + wall["oof_seconds"],
              "historical crop timing identity differs")
    record = verify_record(p, run, digest, summary_checker=check_summary, checkpoint_checker=check_checkpoint)
    record.update(historical_only=True, archive_index_sha256=ARCHIVE_SHA256)
    return record
