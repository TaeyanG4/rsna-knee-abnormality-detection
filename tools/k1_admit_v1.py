"""Strict, offline K1 admission, v1.0.6.

Changelog 1.0.6: compose flat-P validation with the separate A5 contract;
retain v1.0.5 ordinary-A5 and v1.0.4 non-A5 admission output bytes.
Changelog 1.0.5: opt-in A5 config, receipt/Pprime and N32 bindings;
preserve v1.0.4 admission bytes for non-A5 stages.
Changelog 1.0.4: bind flat P to the registered recipe, weights and receipt;
inherit the candidate-bound N32-ratio qualification limit of 9 hours.
Changelog 1.0.3: mandatory candidate-bound N32-ratio record for opt-in stages.
Changelog 1.0.2: interpret an omitted parent drop_o as false; require executed
O members for registered false variants, retaining the mandatory O counter.
Changelog 1.0.1: verify registered drop-O flags/config/member contract before
waiving only the absent O prediction counter; all present counters stay zero.
Changelog 1.0.0: replace candidate-specific admission scripts with fail-closed
receipt, stage, CSV, runtime and ledger checks. Never download or submit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.6"
A5_OUTPUT_VERSION = "1.0.5"
LEGACY_OUTPUT_VERSION = "1.0.4"
LEDGER = ROOT / "artifacts/experiments/SUBMISSION_SCREEN/submit_ledger.jsonl"
TEMPLATE = ROOT / "storage/competition/sample_submission.csv"
OUTPUT = "k1-admission-v1"


class Refuse(ValueError):
    """An admission requirement was not established."""


def need(condition, message):
    if not condition:
        raise Refuse(message)


def local(path):
    p = Path(path)
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    need(p.is_relative_to(ROOT.resolve()), "input/output must stay inside project")
    need(not any(x.lower() in {
        "api-keys", ".ssh", ".kaggle", ".codex", "relabel"
    } or "report" in x.lower() for x in p.parts), "forbidden evidence path")
    return p


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        need(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def decode(data):
    return json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(
                          Refuse("nonfinite JSON constant")))


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


class Evidence:
    def __init__(self):
        self.hashes = {}
        self.checks = []

    def read(self, path):
        p = local(path)
        need(p.is_file(), "missing input: " + str(p))
        data = p.read_bytes()
        digest = sha(data)
        need(str(p) not in self.hashes or self.hashes[str(p)] == digest,
             "input changed during admission: " + str(p))
        self.hashes[str(p)] = digest
        return data

    def obj(self, path):
        obj = decode(self.read(path))
        need(isinstance(obj, dict), "JSON object required: " + str(path))
        return obj

    def check(self, condition, message):
        need(condition, message)
        self.checks.append(message + ": PASS")

    def recheck(self):
        for name, digest in list(self.hashes.items()):
            need(sha(self.read(name)) == digest, "input changed: " + name)


def positive(value, name, zero=False):
    need(type(value) in (float, int) and math.isfinite(value)
         and (value >= 0 if zero else value > 0), "invalid " + name)
    return value


def zero_counters(value, name):
    if isinstance(value, dict):
        need(bool(value), "empty counter: " + name)
        for key, child in value.items():
            zero_counters(child, name + "." + key)
    else:
        need(type(value) in (float, int) and value == 0, "nonzero counter: " + name)


def prediction_fallbacks(record, prefix=""):
    if isinstance(record, list):
        for child in record:
            prediction_fallbacks(child, prefix)
    elif isinstance(record, dict):
        for key, value in record.items():
            name = prefix + "." + key
            if ("prediction" in key and ("fallback" in key or prefix.endswith(".fallbacks"))
                    or key == "r_fallback_count"):
                # Producer boolean assertions are not numerical counters.
                if type(value) is bool and name.endswith(".telemetry.r_fallback_count"):
                    need(value is True, "failed fallback assertion: " + name)
                else:
                    zero_counters(value, name)
            elif isinstance(value, (dict, list)):
                prediction_fallbacks(value, name)


def verified_drop_o(e, stage, reg, k, parent):
    config_path = stage / "config.json"
    config = e.obj(config_path) if config_path.exists() else {}
    records = (reg, config, k, parent)
    # Legacy variants without this contract still require the O counter.
    if not any("drop_o" in record for record in records):
        return False
    need(all(type(record.get("drop_o")) is bool for record in (reg, config, k)),
         "drop_o must be an explicit boolean in stage registration/config and K1 receipt")
    parent_drop_o = parent.get("drop_o", False)
    need(type(parent_drop_o) is bool, "parent drop_o must be a boolean when present")
    drop_o = reg["drop_o"]
    need(all(record["drop_o"] is drop_o for record in (reg, config, k))
         and parent_drop_o is drop_o, "drop_o mismatch")
    config_hash = e.hashes[str(config_path)]
    if is_a5(reg):
        contract = config.get("a5")
        need(isinstance(contract, dict) and contract.get("schema") == "p25-a5-contract/v1"
             and contract == reg.get("a5_contract"), "Registered A5 config contract")
        e.check(sha(encoded(contract)) == reg.get("a5_contract_sha256")
                == k.get("a5_contract_sha256"), "Registered A5 contract SHA256")
        # A5 adds a separate contract to the file; K1 hashes only its own config.
        config_hash = sha(encoded({key: value for key, value in config.items() if key != "a5"}))
        e.check(config_hash == reg.get("registered_config_sha256")
                == k.get("registered_config_sha256"), "Registered A5 inner K1 config SHA256")
    e.check(config_hash == reg.get("config_sha256") == k.get("config_sha256"),
            "Registered drop_o config SHA256 matches K1 receipt")
    if drop_o:
        expected = {
            "skipped_members": {"EXP143v2", "EXP155", "Q_A5q"},
            "intended_members": {"R", "S_R7", "R224"},
            "executed_members": {"R", "S_R7", "R224"},
        }
        for key, members in expected.items():
            value = parent.get(key)
            need(isinstance(value, list) and all(type(v) is str for v in value)
                 and len(value) == len(members) and set(value) == members,
                 "drop_o parent member contract: " + key)
    else:
        executed = parent.get("executed_members")
        need(isinstance(executed, list) and all(type(v) is str for v in executed)
             and {"EXP143v2", "EXP155", "Q_A5q"} <= set(executed),
             "drop_o=false requires executed O members")
    e.checks.append("Explicit consistent drop_o and member contract: PASS")
    return drop_o


def codecs_active(codec):
    need(codec == "ACTIVE" or isinstance(codec, dict)
         and codec.get("status") == "ACTIVE", "codecs must be ACTIVE")
    if isinstance(codec, dict):
        if "s_r7_gate" in codec:
            need(codec["s_r7_gate"] == "ACTIVE", "S_R7 codecs must be ACTIVE")
        if "selftest" in codec:
            need(isinstance(codec["selftest"], dict) and bool(codec["selftest"])
                 and all(v is True for v in codec["selftest"].values()),
                 "codec selftest failed")
        for key in ("hook_errors", "error_samples", "unreadable"):
            need(not codec.get(key), "codec errors: " + key)
        totals = codec.get("totals", {})
        need(isinstance(totals, dict), "invalid codec totals")
        for key in ("plugin_failures", "decode_failures"):
            if key in totals:
                zero_counters(totals[key], "codecs.totals." + key)


def table(data):
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline="")))
    need(len(rows) == 4 and len(rows[0]) == 13 and len(set(rows[0])) == 13,
         "CSV must have 3 rows and 12 unique target columns plus ID")
    need(all(len(row) == 13 for row in rows), "CSV row width")
    need(all(row[0] for row in rows[1:]) and len({r[0] for r in rows[1:]}) == 3,
         "CSV duplicate/empty study IDs")
    for row in rows[1:]:
        for cell in row[1:]:
            try:
                value = float(cell)
            except ValueError:
                raise Refuse("CSV nonnumeric prediction") from None
            need(math.isfinite(value) and 0 <= value <= 1, "CSV predictions must be finite [0,1]")
    return rows[0], [r[0] for r in rows[1:]]


def allowed(record, name):
    for key in ("never_submit", "diagnostic"):
        need(record.get(key, False) is False, name + " is " + key)
    need("n32" not in str(record.get("kernel_id", "")).lower()
         and "n32" not in str(record.get("release_version", "")).lower(),
         name + " is N32")


def ledger_check(e, kernel, version):
    outcomes = []
    for line in e.read(LEDGER).decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = decode(line.encode())
        need(isinstance(row, dict), "malformed ledger record")
        # Any prior attempt also remains the responsibility of submit_once.
        if row.get("kernel") == kernel and str(row.get("version")) == str(version):
            outcomes.append(row.get("outcome"))
    need("SUBMITTED" not in outcomes, "kernel/version already SUBMITTED in ledger")
    need(not any(v in ("IN_FLIGHT", "UNCONFIRMED", "RESOLVED_ABSENT") for v in outcomes),
         "kernel/version has a prior or unresolved attempt in ledger")
    e.checks.append("Local ledger duplicate/prior-attempt check: PASS (no live API query)")


def bound_json(e, base, ref):
    need(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "invalid projection evidence reference")
    p = local(base / ref["path"])
    value = e.obj(p)
    need(e.hashes[str(p)] == ref["sha256"], "projection evidence hash mismatch")
    return value


def is_a5(reg):
    return reg.get("schema") == "p25-a5-registration/v1"


def a5_receipt(e, directory, reg, k, parent, studies, diagnostic):
    a = e.obj(directory / "a5-receipt.json")
    contract = reg.get("a5_contract", {})
    e.check(a.get("schema") == "p25-a5/v1" and a.get("status") == "COMPLETE_A5"
            and a.get("strict_load") is True, "A5 COMPLETE_A5 and strict load")
    e.check(all(type(a.get(key)) is int and a[key] == studies
                for key in ("applicable", "studies")), f"A5 applicable == studies == {studies}")
    e.check(a.get("never_submit") is diagnostic and a.get("diagnostic") is diagnostic,
            "A5 diagnostic/never_submit identity")
    e.check(a.get("dataset") == contract.get("dataset") == "mattiaangeli/knee-mri-fold-weights",
            "A5 registered dataset")
    e.check(isinstance(contract.get("formula"), str) and bool(contract["formula"])
            and a.get("formula") == contract["formula"]
            and reg["formula"].startswith(contract["formula"] + "; inherited outer K1: "),
            "A5 exact registered Pprime formula")
    hashes = contract.get("checkpoint_sha256")
    e.check(isinstance(hashes, list) and len(hashes) == 5
            and all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes)
            and a.get("checkpoint_hashes") == hashes, "A5 registered five checkpoint hashes")
    e.check(k.get("parent_mode") == "P25_A5_PPRIME"
            and k.get("a5_receipt_sha256") == e.hashes[str(directory / "a5-receipt.json")],
            "K1 bound to exact A5 receipt")
    e.check(isinstance(a.get("original_p_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", a["original_p_sha256"])
            and (diagnostic or a["original_p_sha256"] == parent.get("submission_sha256"))
            and isinstance(a.get("pprime_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", a["pprime_sha256"])
            and a["pprime_sha256"] == k.get("parent_sha256"),
            "A5 Pprime -> K1 binding" if diagnostic else "A5 original P -> Pprime -> K1 binding")
    return a


def a5_n32(e, directory, reg, visible):
    need(directory is not None, "A5 requires --n32-dir")
    directory = local(directory)
    n = e.obj(directory / "k1sr-receipt.json")
    e.check(n.get("status") == "COMPLETE_SHARED_K1N" and n.get("k1_executed") is True
            and type(n.get("k1_studies")) is int and n["k1_studies"] == 32,
            "A5 N32 COMPLETE K1 executed 32/32")
    e.check(n.get("kernel_id") == reg["kernel_id"] + "-n32"
            and n.get("release_version") == reg["release_version"] + "-n32"
            and n.get("never_submit") is True and n.get("diagnostic") is True,
            "A5 N32 exact diagnostic sibling")
    e.check(type(n.get("failed_study_count")) is int and n["failed_study_count"] == 0
            and n.get("failed_study_fraction") == 0 and n.get("fail_loud_triggered") is False,
            "A5 N32 zero failed studies")
    for key in ("config_sha256", "registered_config_sha256", "a5_contract_sha256",
                "model_sha256", "formula", "contract", "k1_weight", "parent_weight", "r224_weight"):
        e.check(key in reg and n.get(key) == reg[key] == visible.get(key), "A5 N32 matches " + key)
    if reg.get('p_weight_mode') == 'flat':
        for key in ('p_weight_mode', 'p_raptor_weights'):
            e.check(n.get(key) == reg[key] == visible.get(key), "A5 N32 matches " + key)
    rows = n.get("per_study")
    need(isinstance(rows, list) and len(rows) == 32
         and all(isinstance(r, dict) and r.get("ok") is True
                 and isinstance(r.get("uid"), str) and r["uid"]
                 and r.get("config_sha256") == reg["config_sha256"] for r in rows)
         and len({r["uid"] for r in rows}) == 32, "A5 N32 32 unique successful bound studies")
    p = e.obj(directory / "c85r7-receipt.json")
    e.check(p.get("status") == "DIAGNOSTIC_C85R7" and p.get("version") == n["release_version"]
            and p.get("never_submit") is True and p.get("diagnostic") is True
            and type(p.get("studies")) is int and p["studies"] == 32 and p.get("findings") == 12,
            "A5 N32 parent DIAGNOSTIC 32 x 12")
    prediction_fallbacks(p)
    codecs_active(p.get("codecs"))
    a = a5_receipt(e, directory, reg, n, p, 32, True)
    versions = set()
    for name in ("k1sr-receipt.json", "c85r7-receipt.json", "a5-receipt.json"):
        d = e.obj(directory / (name + ".download.json"))
        e.check(d.get("status") == "DOWNLOADED" and d.get("kernel") == n["kernel_id"]
                and type(d.get("version")) is int and d["version"] > 0
                and d.get("file") == d.get("local_name") == name
                and d.get("sha256") == sha(e.read(directory / name)), "A5 N32 download provenance " + name)
        versions.add(d["version"])
    need(len(versions) == 1, "A5 N32 download versions mismatch")
    seconds = positive(a.get("seconds"), "A5 N32 seconds")
    e.checks.append(f"A5 N32 execution verified; A5 seconds={seconds}; no hidden runtime projection claimed: PASS")


def projection(e, directory, visible):
    n = e.obj(directory / "k1sr-receipt.json")
    e.check(isinstance(n.get("status"), str) and n["status"].startswith("COMPLETE_SHARED_K1")
            and n.get("k1_executed") is True and type(n.get("k1_studies")) is int
            and n["k1_studies"] == 32, "N32 COMPLETE K1 executed 32/32")
    rows = n.get("per_study")
    need(isinstance(rows, list) and len(rows) == 32 and
         all(isinstance(r, dict) and r.get("ok") is True and isinstance(r.get("uid"), str) for r in rows)
         and len({r["uid"] for r in rows}) == 32, "N32 successful unique per-study rows required")
    for key in ("model_sha256", "formula", "contract", "r224_weight"):
        e.check(key in visible and n.get(key) == visible[key], "N32 matches visible " + key)
    spec = e.obj(directory / "projection-inputs.json")
    need(spec.get("schema") == "k1-admit-projection/v1", "unsupported projection rule")
    receipt_hash = e.hashes[str(local(directory / "k1sr-receipt.json"))]
    need(spec.get("receipt_sha256") == receipt_hash, "projection bound to different N32 receipt")
    need(isinstance(spec.get("compatibility_reason"), str) and spec["compatibility_reason"].strip(),
         "document N32/visible runtime compatibility")
    startup = bound_json(e, directory, spec.get("startup"))
    selection = bound_json(e, directory, spec.get("selection"))
    train = bound_json(e, directory, spec.get("train"))
    need(startup.get("receipt_sha256") == receipt_hash
         and isinstance(startup.get("method"), str) and startup["method"].strip(),
         "startup measurement must name N32 receipt and method")
    need(isinstance(train.get("method"), str) and train["method"].strip(),
         "training DICOM count provenance required")
    counts = selection.get("counts")
    need(selection.get("n") == 32 and isinstance(counts, dict)
         and set(counts) == {r["uid"] for r in rows}, "N32 roster/counts mismatch")
    roster_total = positive(sum(positive(v, "roster DICOM count") for v in counts.values()),
                            "roster DICOM total")
    roster_mean = roster_total / 32
    train_files = positive(train.get("dicom_files"), "train DICOM count")
    train_studies = positive(train.get("studies"), "train study count")
    train_mean = positive(train_files / train_studies, "train DICOM mean")
    wall = positive(n.get("parent_seconds"), "parent pipeline wall")
    start = positive(startup.get("parent_startup_seconds"), "parent startup", zero=True)
    need(start < wall, "startup must be smaller than measured parent wall")
    phases = n.get("phases", {})
    need(isinstance(phases, dict), "N32 phases object required")
    classify = positive(phases.get("classifier_seconds"), "K1 classify")
    extra = positive(phases.get("extra_render_seconds"), "K1 extra render", zero=True)
    load = positive(phases.get("load_seconds"), "K1 load", zero=True)
    for key, aggregate in (("classifier_seconds", classify),
                           ("extra_render_seconds", extra), ("load_seconds", load)):
        total = sum(positive(r.get(key), "N32 per-study " + key, zero=True) for r in rows)
        need(math.isclose(total, aggregate, rel_tol=1e-6, abs_tol=1e-6),
             "N32 phase/per-study timing mismatch: " + key)
    parent_per = (wall - start) / 32 * train_mean / roster_mean
    k1_per = (classify + extra) / 32
    hours = (start + load + 1300 * (parent_per + k1_per)) / 3600
    need(all(math.isfinite(v) for v in (parent_per, k1_per, hours)),
         "nonfinite derived runtime projection")
    result = dict(rule="(startup + K1_load + 1300*((parent_wall-startup)/32*"
                       "(train_DICOM/train_studies)/(roster_DICOM/32)"
                       "+(K1_classify+K1_extra_render)/32))/3600",
                  parent_wall_seconds=wall, parent_startup_seconds=start,
                  k1_classify_seconds=classify, k1_extra_render_seconds=extra,
                  k1_load_seconds=load, n32_studies=32, hidden_studies=1300,
                  roster_dicom_files=roster_total, train_dicom_files=train_files,
                  train_studies=train_studies, startup_method=startup["method"],
                  train_count_method=train["method"],
                  roster_mean_dicom=roster_mean, train_mean_dicom=train_mean,
                  parent_per_study_seconds=parent_per, k1_per_study_seconds=k1_per,
                  projected_hours=hours, limit_hours=8.0,
                  compatibility_reason=spec["compatibility_reason"],
                  scope="Parent wall retains in-pass K1 overlap (conservative double counting). "
                        "Only parent steady-state is roster-corrected; startup/load counted once. "
                        "Estimate, not a runtime guarantee.")
    print("PROJECTION " + json.dumps(result, sort_keys=True))
    e.check(hours <= 8.0, "N32 1300-study projection <= 8.0 hours")
    return result


def validate(stage, kernel, version, visible_dir, n32_dir=None):
    e = Evidence()
    stage, visible = local(stage), local(visible_dir)
    need(re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", kernel), "invalid owner/slug")
    need(type(version) is int and version > 0, "version must be a positive integer")
    need("n32" not in kernel.lower() and "n32" not in stage.name.lower(), "N32 candidate prohibited")
    k = e.obj(visible / "k1sr-receipt.json")
    e.check(isinstance(k.get("status"), str) and k["status"].startswith("COMPLETE_SHARED_K1")
            and k.get("k1_executed") is True, "K1 COMPLETE_SHARED_K1 and executed (no degraded parent-only)")
    e.check(type(k.get("k1_studies")) is int and k["k1_studies"] == 3, "K1 studies == 3")
    allowed(k, "K1 receipt")
    reg = e.obj(stage / "registration.json")
    allowed(reg, "Stage")
    meta = e.obj(stage / "kernel/kernel-metadata.json")
    e.check(k.get("kernel_id") == reg.get("kernel_id") == meta.get("id") == kernel,
            "Exact receipt/stage/metadata kernel identity")
    release = stage.name.removeprefix("v")
    e.check(k.get("release_version") == reg.get("release_version") == release, "Stage release version")
    e.check(local(reg.get("stage", "")) == stage, "Registered stage directory")
    for filename, key in (("release.ipynb", "kernel_sha256"), ("kernel-metadata.json", "metadata_sha256")):
        e.check(sha(e.read(stage / "kernel" / filename)) == reg.get(key), "Registered " + filename + " hash")
    formula = k.get("formula")
    e.check(isinstance(formula, str) and bool(formula.strip()) and "\n" not in formula
            and "\r" not in formula and formula == reg.get("formula"), "Exact receipt/stage formula")
    for key in ("model_sha256", "contract", "k1_weight", "parent_weight", "r224_weight"):
        e.check(key in reg and k.get(key) == reg[key], "Receipt/stage " + key)
    parent = e.obj(visible / "c85r7-receipt.json")
    e.check(parent.get("status") == "COMPLETE_C85R7" and parent.get("version") == release,
            "Parent COMPLETE_C85R7 and matching release")
    e.check(parent.get("studies") == 3 and parent.get("findings") == 12, "Parent 3 studies x 12 findings")
    drop_o = verified_drop_o(e, stage, reg, k, parent)
    required = {"r_prediction", "s_r7_prediction"}
    if not drop_o:
        required.add("a172_prediction")
    need(isinstance(parent.get("fallbacks"), dict) and
         required <= parent["fallbacks"].keys(),
         "missing parent prediction fallback counters")
    if "R224" in parent.get("intended_members", []):
        need("r224_prediction" in parent["fallbacks"], "missing R224 prediction fallback counter")
    prediction_fallbacks(parent)
    e.checks.append("All prediction fallback counters zero (anatomy anchors excluded): PASS")
    codecs_active(parent.get("codecs"))
    e.checks.append("Codecs ACTIVE: PASS")
    parent_bytes = e.read(visible / "k1sr-parent.csv")
    if is_a5(reg):
        a = a5_receipt(e, visible, reg, k, parent, 3, False)
        e.check(sha(parent_bytes) == k.get("parent_sha256") == a["pprime_sha256"],
                "A5 Pprime CSV SHA256 matches A5 and K1 receipts")
    else:
        e.check(sha(parent_bytes) == k.get("parent_sha256") == parent.get("submission_sha256"),
                "Parent CSV SHA256 matches both receipts")
    submission = e.read(visible / "submission.csv")
    e.check(sha(submission) == k.get("submission_sha256"), "Final submission CSV SHA256")
    template = table(e.read(TEMPLATE))
    e.check(table(submission) == template and table(parent_bytes) == template,
            "Submission/parent exactly 3 template IDs x 12 finite [0,1] targets in template order")
    # The release receipt carries a release version, not a Kaggle version.
    # Exact-version download receipts supply that otherwise missing provenance.
    download_names = ("k1sr-receipt.json", "c85r7-receipt.json", "submission.csv")
    if is_a5(reg):
        download_names += ("a5-receipt.json",)
    for name in download_names:
        download = e.obj(visible / (name + ".download.json"))
        e.check(download.get("status") == "DOWNLOADED" and download.get("kernel") == kernel
                and type(download.get("version")) is int and download["version"] == version
                and download.get("file") == name and download.get("local_name") == name
                and download.get("sha256") == sha(e.read(visible / name)),
                "Exact-version download provenance " + name)
    config = e.obj(stage / "config.json") if (stage / "config.json").exists() else {}
    if any('p_weight_mode' in obj or 'p_raptor_weights' in obj for obj in (config, reg, k)):
        from tools import k1n_stage_v1 as builder
        inner = {key: value for key, value in config.items() if key != 'a5'} if is_a5(reg) else config
        expected = builder.config(builder.canonical(inner))
        if is_a5(reg):
            expected['formula'] = reg['a5_contract']['formula'] + '; inherited outer K1: ' + expected['formula']
        need(expected.get('p_weight_mode') == 'flat', 'registered flat P recipe required')
        for name, obj in (('stage', reg), ('visible', k)):
            for key in ('p_weight_mode', 'p_raptor_weights', 'config_sha256',
                        'registered_config_sha256', 'formula'):
                e.check(key in obj and obj[key] == expected[key],
                        name + ' registered flat P ' + key)
    if is_a5(reg):
        a5_n32(e, n32_dir, reg, k)
        projected = None
    elif 'projection_basis' in config or 'projection_basis' in reg:
        need(config.get('projection_basis') == reg.get('projection_basis') == 'n32_ratio',
             'registered projection_basis mismatch')
        from tools import k1n_qualify_v1
        projected = k1n_qualify_v1.verify(e, stage, n32_dir)
    else:
        projected = projection(e, local(n32_dir), k) if n32_dir is not None else None
    if projected is None and not is_a5(reg):
        e.checks.append("N32: NOT PROVIDED (optional; no runtime qualification claimed)")
    ledger_check(e, kernel, version)
    e.recheck()
    return e, reg, k, projected


def write_once(path, data):
    """Immutable output; no replacement/deletion of existing admissions."""
    path = local(path)
    if path.exists():
        need(path.read_bytes() == data, "existing admission differs; preserve it and use a fresh visible copy: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(data)


def command_line(args):
    # A PowerShell-safe one-liner; never interpolated into a shell by this tool.
    return "& " + " ".join("'" + str(arg).replace("'", "''") + "'" for arg in args)


def prepare(stage, kernel, version, visible_dir, n32_dir=None):
    e, reg, k, projected = validate(stage, kernel, version, visible_dir, n32_dir)
    visible, stage = local(visible_dir), local(stage)
    out = local(visible / OUTPUT)
    out.mkdir(exist_ok=True)
    output_version = (VERSION if reg.get('p_weight_mode') == 'flat' else A5_OUTPUT_VERSION) if is_a5(reg) else LEGACY_OUTPUT_VERSION
    registration = dict(
        schema="ordinary-visible-submission-registration/v1", tool_version=output_version,
        kernel_id=kernel, version=k["release_version"], stage=str(stage),
        kernel_sha256=reg["kernel_sha256"], metadata_sha256=reg["metadata_sha256"],
        stage_registration_sha256=e.hashes[str(stage / "registration.json")],
        formula=k["formula"], execution_verified=True, input_sha256=dict(e.hashes),
        projection=projected, independent_review="NOT PERFORMED BY THIS TOOL",
    )
    rp = out / "ordinary-visible-registration.json"
    registration_bytes = encoded(registration)
    binding = dict(schema_version="1.0.0", kernel=kernel, kernel_version=version,
                   receipt_sha256=e.hashes[str(visible / "c85r7-receipt.json")],
                   registration=str(rp), registration_sha256=sha(registration_bytes),
                   k1_receipt_sha256=e.hashes[str(visible / "k1sr-receipt.json")],
                   submission_sha256=k["submission_sha256"])
    review = (f"# K1 automated admission v{output_version}\n\n"
              f"Kernel: {kernel}\nKaggle version: {version}\nStage: {stage}\n\n"
              f"Formula (verbatim receipt): {k['formula']}\n\n"
              "Machine checks only; this document is not an independent reviewer verdict.\n"
              "B must retain the independent review and refresh live allowance/duplicates before submission.\n\n"
              + "\n".join("- " + check for check in e.checks)
              + "\n\nProjection: " + json.dumps(projected, sort_keys=True)
              + "\n\nVERDICT: PASS\n").encode()
    write_once(rp, registration_bytes)
    write_once(out / "submission-binding.json", encoded(binding))
    # Canonical wrapper binds its sibling sidecar. Original receipts stay untouched.
    write_once(out / "c85r7-receipt.json", e.read(visible / "c85r7-receipt.json"))
    pending = out / "machine-check.pending.md"
    write_once(pending, review)
    for path in (rp, out / "submission-binding.json", out / "c85r7-receipt.json", pending):
        e.read(path)
    identity = "K1-" + sha(f"{kernel}:{version}".encode())[:20]
    base = [sys.executable, str(ROOT / "tools/kaggle_submit_once.py"),
            "--id", identity, "--kernel", kernel, "--version", str(version),
            "--message", k["formula"], "--receipt", str(out / "c85r7-receipt.json")]
    result = subprocess.run(base + ["--review", str(pending), "--check-only"],
                            cwd=ROOT, capture_output=True, text=True, check=False)
    print("CHECK-ONLY " + result.stdout.strip())
    need(result.returncode == 0, "kaggle_submit_once --check-only REFUSED")
    try:
        outcome = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise Refuse("check-only returned malformed outcome") from None
    need(isinstance(outcome, dict) and outcome.get("outcome") == "ADMITTED_OFFLINE",
         "check-only did not admit")
    e.recheck()
    ledger_check(e, kernel, version)
    final_review = out / "READY-REVIEW.md"
    write_once(final_review, review.replace(
        b"\n\nVERDICT: PASS", b"\n\nCanonical --check-only: ADMITTED_OFFLINE\n\nVERDICT: PASS"))
    print("PASS: " + str(out))
    command = command_line(base + ["--review", str(final_review)])
    print("SUBMIT COMMAND (NOT EXECUTED): " + command)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--stage", required=True)
    p.add_argument("--kernel", required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--visible-dir", required=True)
    p.add_argument("--n32-dir")
    args = vars(parser.parse_args(argv))
    args.pop("action")
    try:
        prepare(**args)
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        print("REFUSE: " + str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
