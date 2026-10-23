"""Shared arm contracts, v1.3.0 (2026-10-11, astra).

Changelog: explicit BN-free evidence for architectures with zero BN modules.
v1.3.0: explicit production member specs bind a declared target and evidence,
without controls. Historical experimental specs retain their strict contract.
Import is stdlib-only and has no side effects. File hashes establish integrity,
not authorship or proof of execution: the reviewer must trust evidence producers.
Artifact receipts do not grant scientific acceptance or execution permission.
No wall limits: elapsed time is measurement only. Family runners must supply a
fresh target observation at execution; this module never launches commands.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tarfile
import uuid


VERSION = "1.3.0"
ENVIRONMENT_SCHEMA = "arm-target-environment/v1"
MEMBER_ENVIRONMENT_SCHEMA = "arm-member-target/v1"
VERIFICATION_SCHEMA = "arm-verification/v1"
RUNTIME_KEYS = frozenset({
    "python", "driver", "torch", "cuda", "cudnn", "gpu", "timm", "numpy", "pydicom", "cv2",
})
PROVENANCE_RUNTIME_KEYS = RUNTIME_KEYS - {"python", "driver"}
BINDING_KEYS = frozenset({"sources", "config", "data", "rosters", "controls", "environment"})
SMOKE_COUNTS = frozenset({"train_batches", "validation_batches", "bn_batches", "optimizer_steps"})
SMOKE_FLAGS = frozenset({"real_data", "checkpoint_reloaded", "finite_outputs"})


class ContractError(ValueError):
    """A missing, malformed, stale, or mismatched arm contract."""


def _need(condition, message):
    if not condition:
        raise ContractError(message)


def _keys(value, keys, name):
    _need(type(value) is dict and set(value) == set(keys), f"{name}: fields differ")


def _text(value, name):
    _need(type(value) is str and bool(value.strip()), f"{name}: nonempty string required")


def _hash(value, name):
    _need(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
          f"{name}: lowercase SHA256 required")


def _hash_map(value, name):
    _need(type(value) is dict and bool(value), f"{name}: nonempty hash map required")
    for key, digest in value.items():
        _text(key, name)
        _hash(digest, name)


def _json_value(value):
    if type(value) is dict:
        _need(all(type(key) is str for key in value), "JSON object keys must be strings")
        for item in value.values():
            _json_value(item)
    elif type(value) is list:
        for item in value:
            _json_value(item)
    elif type(value) is float:
        _need(math.isfinite(value), "JSON numbers must be finite")
    else:
        _need(value is None or type(value) in (str, bool, int), "unsupported JSON value")


def canonical_bytes(value):
    """Canonical UTF-8 JSON, preserving list order and exact numeric types."""
    _json_value(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def content_sha256(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def parse_json(data):
    """Reject duplicate fields and nonfinite numbers rather than silently coerce."""
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            _need(key not in result, "duplicate JSON field")
            result[key] = value
        return result

    def reject_constant(_):
        raise ContractError("JSON numbers must be finite")

    try:
        value = json.loads(data, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    except (ValueError, TypeError, UnicodeError) as exc:
        # Do not echo payloads: they may contain private identifiers or labels.
        raise ContractError("invalid strict JSON") from exc
    _json_value(value)
    return value


def validate_environment(profile):
    _keys(profile, {"schema", "hostname", "runtime", "evidence"}, "target environment")
    _need(profile["schema"] in (ENVIRONMENT_SCHEMA, MEMBER_ENVIRONMENT_SCHEMA), "target environment: schema differs")
    _text(profile["hostname"], "target hostname")
    _keys(profile["runtime"], RUNTIME_KEYS, "target runtime")
    for key, value in profile["runtime"].items():
        if key == "cudnn":
            _need(type(value) is int and value > 0, "cudnn: positive integer required")
        else:
            _text(value, f"runtime {key}")
    names = {"declaration.json"} if profile["schema"] == MEMBER_ENVIRONMENT_SCHEMA else {"summary.json", "provenance.json"}
    _keys(profile["evidence"], names, "target evidence")
    _hash_map(profile["evidence"], "target evidence")


def environment_from_control(summary_path, provenance_path, *, expected_hashes):
    """Extract a target profile from a previously qualified control's records.

    expected_hashes is the summary/provenance subset of the collected control
    receipt, supplied by the caller from its trusted manifest. Registration
    environments are deliberately not an input. Hash and parse the same bytes.
    Python/driver come from summary; the eight legacy runtime fields must also
    agree with provenance. No defaults and no unrecorded local observations.
    """
    _keys(expected_hashes, {"summary.json", "provenance.json"}, "control hashes")
    _hash_map(expected_hashes, "control hashes")
    documents = {}
    for name, path in (("summary.json", summary_path), ("provenance.json", provenance_path)):
        raw = Path(path).read_bytes()
        _need(hashlib.sha256(raw).hexdigest() == expected_hashes[name], "control evidence hash differs")
        documents[name] = parse_json(raw)
    summary = documents["summary.json"]
    provenance = documents["provenance.json"]
    _need(type(summary) is dict and type(provenance) is dict, "control documents must be objects")
    recorded = summary.get("environment")
    _need(type(recorded) is dict and RUNTIME_KEYS | {"hostname"} <= set(recorded),
          "control summary: recorded runtime incomplete")
    legacy = provenance.get("environment")
    _keys(legacy, PROVENANCE_RUNTIME_KEYS, "control provenance runtime")
    _need(canonical_bytes(legacy) == canonical_bytes({key: recorded[key] for key in legacy}),
          "control summary/provenance runtime differs")
    profile = {"schema": ENVIRONMENT_SCHEMA, "hostname": recorded["hostname"],
               "runtime": {key: recorded[key] for key in RUNTIME_KEYS},
               "evidence": dict(expected_hashes)}
    validate_environment(profile)
    return profile


def environment_from_declaration(path, *, expected_sha256):
    """Bind a declared production target, not a fresh observation or a control.

    Execution still requires a fresh observation and independent PASS with
    actual target smoke evidence. The declaration grants no launch authority.
    """
    _hash(expected_sha256, "target declaration")
    raw = Path(path).read_bytes()
    _need(hashlib.sha256(raw).hexdigest() == expected_sha256, "target declaration hash differs")
    declared = parse_json(raw)
    _keys(declared, {"hostname", "runtime", "backend_policy"}, "target declaration")
    _need(type(declared["backend_policy"]) is dict and bool(declared["backend_policy"]),
          "target backend policy required")
    profile = dict(schema=MEMBER_ENVIRONMENT_SCHEMA, hostname=declared["hostname"],
                   runtime=declared["runtime"], evidence={"declaration.json": expected_sha256})
    validate_environment(profile)
    return profile, declared


def check_environment(profile, *, hostname, runtime):
    """Compare fresh target-host observations; never a registration environment."""
    validate_environment(profile)
    _need(hostname == profile["hostname"], "target hostname differs")
    _keys(runtime, RUNTIME_KEYS, "observed runtime")
    changed = sorted(key for key in RUNTIME_KEYS
                     if canonical_bytes(runtime[key]) != canonical_bytes(profile["runtime"][key]))
    _need(not changed, "target runtime differs: " + ", ".join(changed))


def validate_bindings(bindings):
    member = type(bindings) is dict and "evidence" in bindings
    _keys(bindings, BINDING_KEYS | ({"evidence"} if member else set()), "verification bindings")
    for key in bindings:
        if key in {"config", "environment"}:
            _hash(bindings[key], f"binding {key}")
        elif member and key == "controls":
            _need(bindings[key] == {}, "member controls must be empty")
        else:
            _hash_map(bindings[key], f"binding {key}")


def _command_result(record, *, extra_keys, name):
    _keys(record, {"command", "exit_code", "log_sha256"} | set(extra_keys), name)
    _need(type(record["command"]) is list and bool(record["command"]), f"{name}: command required")
    for arg in record["command"]:
        _text(arg, f"{name} command argument")
    _need(type(record["exit_code"]) is int, f"{name}: integer exit code required")
    _hash(record["log_sha256"], f"{name} log")


def _validate_smoke(smoke, kind, bindings, target, bn_none=False):
    _command_result(smoke, extra_keys={"hostname", "device", "environment_sha256",
                                     "bindings_sha256", "results"}, name=f"{kind} smoke")
    _text(smoke["hostname"], "smoke hostname")
    _hash(smoke["environment_sha256"], "smoke environment")
    _hash(smoke["bindings_sha256"], "smoke bindings")
    _need(smoke["bindings_sha256"] == content_sha256(bindings), "smoke bindings differ")
    _need(smoke["device"] == ("cpu" if kind == "cpu" else "cuda"), "smoke device differs")
    if kind == "target":
        _need(smoke["environment_sha256"] == bindings["environment"] and
              smoke["hostname"] == target["hostname"], "smoke target environment differs")
    results = smoke["results"]
    _keys(results, SMOKE_COUNTS | SMOKE_FLAGS | ({"bn_module_count"} if bn_none else set()), "smoke results")
    if bn_none:
        _need(type(results["bn_module_count"]) is int and results["bn_module_count"] == 0
              and results["bn_batches"] == 0, "BN-free smoke must observe zero BN modules and batches")
    for key in SMOKE_COUNTS:
        _need(type(results[key]) is int and results[key] >= 0, "smoke count must be a nonnegative integer")
    for key in SMOKE_FLAGS:
        _need(type(results[key]) is bool, "smoke flag must be boolean")


def validate_verification_note(note, *, expected_scope, expected_bindings, target_environment, expected_bn=None):
    """Validate an arm note against independently supplied identities.

    PASS requires both real-data CPU and target-GPU smoke receipts, including
    actual train/validation/BN batches, optimizer updates and checkpoint reload.
    HOLD/CHALLENGED may record partial or failed work. This checks declarations;
    staging/collect must separately verify the referenced logs and artifacts.
    """
    bn_none = expected_bn is not None
    if bn_none:
        _need(canonical_bytes(expected_bn) == canonical_bytes({"policy": "none", "module_count": 0}),
              "unsupported BN-free contract")
    validate_environment(target_environment)
    validate_bindings(expected_bindings)
    _need(expected_bindings["environment"] == content_sha256(target_environment),
          "expected environment binding differs")
    _text(expected_scope, "expected scope")
    _keys(note, {"schema", "scope", "reviewer", "verdict", "blockers", "notes",
                 "bindings", "checks", "smokes"}, "verification note")
    _need(note["schema"] == VERIFICATION_SCHEMA, "verification schema differs")
    _need(note["scope"] == expected_scope, "verification scope differs")
    _keys(note["reviewer"], {"name", "host"}, "reviewer")
    for value in note["reviewer"].values():
        _text(value, "reviewer identity")
    _need(note["verdict"] in ("PASS", "HOLD", "CHALLENGED"), "verification verdict differs")
    for key in ("blockers", "notes"):
        _need(type(note[key]) is list, f"{key}: list required")
        for value in note[key]:
            _text(value, key)
    validate_bindings(note["bindings"])
    _need(canonical_bytes(note["bindings"]) == canonical_bytes(expected_bindings),
          "verification bindings differ")
    _need(type(note["checks"]) is list and bool(note["checks"]), "verification checks required")
    names = set()
    for check in note["checks"]:
        _command_result(check, extra_keys={"name"}, name="verification check")
        _text(check["name"], "check name")
        _need(check["name"] not in names, "duplicate check name")
        names.add(check["name"])
    _keys(note["smokes"], {"cpu", "target"}, "smokes")
    for kind, smoke in note["smokes"].items():
        if smoke is not None:
            _validate_smoke(smoke, kind, expected_bindings, target_environment, bn_none)
    if note["verdict"] == "PASS":
        _need(not note["blockers"], "PASS has blockers")
        _need(all(check["exit_code"] == 0 for check in note["checks"]), "PASS has a failed check")
        for kind, smoke in note["smokes"].items():
            _need(smoke is not None, f"PASS requires real-data {kind} smoke")
            _need(smoke["exit_code"] == 0 and all(smoke["results"][key] for key in SMOKE_FLAGS)
                  and all(smoke["results"][key] > 0 for key in SMOKE_COUNTS - ({"bn_batches"} if bn_none else set())),
                  f"PASS requires successful real-data {kind} smoke")


def require_verification_pass(note, **expected):
    """Return the exact note identity after validation and an exact PASS verdict."""
    validate_verification_note(note, **expected)
    _need(note["verdict"] == "PASS", "verification is not PASS")
    return content_sha256(note)


SPEC_SCHEMA = "arm-spec/v1"
MEMBER_SPEC_SCHEMA = "arm-member-spec/v1"
RECEIPT_SCHEMA = "arm-artifact-receipt/v1"
EVIDENCE_SCHEMA = "arm-command-evidence/v1"
FILE_GROUPS = ("sources", "data", "rosters", "controls")
CONFIG_KEYS = frozenset({"model", "labels", "targets", "windows", "sampler", "updates",
                         "bn", "readout", "backend_policy", "roots"})
STAGE_METADATA = frozenset({"arm-spec.json", "stage-receipt.json"})


def safe_member(name):
    """Portable regular-file name; reject Windows aliases as well as traversal."""
    _text(name, "member")
    parts = name.split("/")
    _need(all(part and part not in (".", "..") and not part.endswith((".", " "))
              and not re.search(r'[<>:"\\|?*\x00-\x1f]', part)
              and not re.fullmatch(r"(?i:con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\..*)?", part)
              for part in parts), "unsafe member path")
    return name


def _inventory(value):
    _hash_map(value, "file inventory")
    seen = set()
    for name in value:
        safe_member(name)
        _need(name.casefold() not in seen, "case-colliding members")
        seen.add(name.casefold())
    _need(not any("/".join(name.split("/")[:i]).casefold() in seen
                  for name in value for i in range(1, len(name.split("/")))),
          "file/directory member collision")


def _path(root, member):
    safe_member(member)
    _need(not _linked(Path(root)), "linked root forbidden")
    root = Path(root).resolve()
    result = root / member
    current = root
    for part in member.split("/"):
        current /= part
        _need(not _linked(current), "linked member forbidden")
    _need(result.resolve().is_relative_to(root), "member escapes root")
    return result


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _linked(path):
    # pathlib.is_junction exists on the local 3.12 but not the Vast 3.11 host.
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _verify_files(root, inventory):
    _inventory(inventory)
    for member, digest in inventory.items():
        path = _path(root, member)
        _need(path.is_file() and file_sha256(path) == digest, "file missing or hash differs: " + member)


def _files(root):
    root = Path(root)
    _need(root.is_dir() and not _linked(root), "regular directory required")
    names = set()
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(base) / name
            _need(not _linked(path), "linked member forbidden")
        names.update((Path(base) / name).relative_to(root).as_posix() for name in files)
    return names


@dataclass(frozen=True)
class ArmSpec:
    """Canonical bytes prevent nested dict mutation after registration.

    Family code validates scientific values; this layer requires their explicit
    presence and binds them all. config is the sole family contract, never a
    Python import, generated source, or permissive command-line override.
    """
    encoded: bytes

    def __post_init__(self):
        doc = parse_json(self.encoded)
        member = type(doc) is dict and doc.get("schema") == MEMBER_SPEC_SCHEMA
        _keys(doc, {"schema", "family", "arm", "seed", "fold", "attempt", "config",
                    "inputs", "target_environment", "outputs"} | ({"run_kind"} if member else set()), "arm spec")
        _need(doc["schema"] in (SPEC_SCHEMA, MEMBER_SPEC_SCHEMA), "spec schema differs")
        if member:
            _need(doc["run_kind"] == "member" and doc["family"] == "convnext-mil", "unsupported member run kind/family")
        _need(doc["family"] in ("coatnet-bridge", "convnext-mil"), "unknown arm family")
        for key in ("arm", "attempt"):
            safe_member(doc[key])
            _need("/" not in doc[key], "identity must be one component")
        _need(type(doc["seed"]) is int and doc["seed"] >= 0, "invalid seed")
        _need(type(doc["fold"]) is int and 0 <= doc["fold"] <= 4, "invalid fold")
        if doc["family"] == "coatnet-bridge":
            _need(doc["seed"] == 42 and doc["fold"] == 0, "bridge requires seed42/fold0")
        _keys(doc["config"], CONFIG_KEYS, "scientific config")
        for key in CONFIG_KEYS - {"targets"}:
            _need(type(doc["config"][key]) is dict and bool(doc["config"][key]),
                  "explicit scientific contract required: " + key)
        targets = doc["config"]["targets"]
        _need(type(targets) is list and len(targets) == 12 and
              all(type(t) is str and t.strip() for t in targets) and len(set(targets)) == 12,
              "twelve ordered unique targets required")
        groups = (*FILE_GROUPS, "evidence") if member else FILE_GROUPS
        _keys(doc["inputs"], groups, "input groups")
        if member:
            _need(type(doc["inputs"]["evidence"]) is dict and len(doc["inputs"]["evidence"]) == 2,
                  "member requires recipe and target evidence")
        combined = {}
        for group in groups:
            if member and group == "controls":
                _need(doc["inputs"][group] == {}, "member controls must be empty")
                continue
            _inventory(doc["inputs"][group])
            _need(not set(combined) & set(doc["inputs"][group]), "duplicate input member")
            combined.update(doc["inputs"][group])
        _inventory(combined)
        _need(not {p.casefold() for p in combined} & STAGE_METADATA, "reserved stage member")
        _inventory({**combined, **dict.fromkeys(STAGE_METADATA, "0" * 64)})
        validate_environment(doc["target_environment"])
        _need(doc["target_environment"]["schema"] ==
              (MEMBER_ENVIRONMENT_SCHEMA if member else ENVIRONMENT_SCHEMA), "run kind target schema differs")
        _need(type(doc["outputs"]) is list and bool(doc["outputs"]), "required outputs missing")
        for name in doc["outputs"]:
            safe_member(name)
        _need(len(set(doc["outputs"])) == len(doc["outputs"]), "duplicate output")
        _inventory({name: "0" * 64 for name in doc["outputs"]})
        _need(not {p.casefold() for p in doc["outputs"]} & {"receipt.json", "collection.json"},
              "reserved output member")
        _inventory(dict.fromkeys(doc["outputs"] + ["receipt.json", "collection.json"], "0" * 64))
        object.__setattr__(self, "encoded", canonical_bytes(doc))

    @classmethod
    def from_document(cls, document):
        return cls(canonical_bytes(document))

    @property
    def document(self):
        return parse_json(self.encoded)

    @property
    def sha256(self):
        return hashlib.sha256(self.encoded).hexdigest()

    @property
    def run_kind(self):
        return self.document.get("run_kind", "experiment")

    @property
    def scope(self):
        d = self.document
        prefix = "member/" if self.run_kind == "member" else ""
        return prefix + f"{d['family']}/{d['arm']}/seed{d['seed']}/fold{d['fold']}/{d['attempt']}"

    @property
    def bindings(self):
        d = self.document
        return {**d["inputs"], "config": self.sha256,
                "environment": content_sha256(d["target_environment"])}

    @property
    def inventory(self):
        return {name: digest for group in self.document["inputs"].values() for name, digest in group.items()}

    @property
    def verification_args(self):
        result = {"expected_scope": self.scope, "expected_bindings": self.bindings,
                  "target_environment": self.document["target_environment"]}
        if self.document["config"]["bn"].get("policy") == "none":
            result["expected_bn"] = self.document["config"]["bn"]
        return result


def create_new(path, data):
    """Exclusive creation; never silently replace a registration or receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(data)


def register(document, path):
    spec = ArmSpec.from_document(document)
    create_new(path, spec.encoded)
    return spec


def validate_inputs(spec, root):
    _verify_files(root, spec.inventory)
    profile = spec.document["target_environment"]
    if spec.run_kind == "member":
        digest = profile["evidence"]["declaration.json"]
        matches = [p for p, h in spec.document["inputs"]["evidence"].items() if h == digest]
        _need(len(matches) == 1, "target declaration must have one bound evidence member")
        observed, declaration = environment_from_declaration(_path(root, matches[0]), expected_sha256=digest)
        _need(canonical_bytes(observed) == canonical_bytes(profile), "target profile differs from declaration")
        check_target_observation(spec, declaration)
        recipe_path = next(p for p in spec.document["inputs"]["evidence"] if p != matches[0])
        recipe = ArmSpec(_path(root, recipe_path).read_bytes()).document
        doc = spec.document
        for key in ("family", "seed", "fold", "config", "outputs"):
            _need(canonical_bytes(doc[key]) == canonical_bytes(recipe[key]), "member recipe differs: " + key)
        for key in ("data", "rosters"):
            _need(canonical_bytes(doc["inputs"][key]) == canonical_bytes(recipe["inputs"][key]),
                  "member recipe inputs differ: " + key)
        return
    controls = spec.document["inputs"]["controls"]
    records = {}
    for name, digest in profile["evidence"].items():
        matches = [member for member, value in controls.items() if value == digest]
        _need(len(matches) == 1, "target record must have one bound control member")
        records[name] = _path(root, matches[0])
    observed = environment_from_control(records["summary.json"], records["provenance.json"],
                                        expected_hashes=profile["evidence"])
    _need(canonical_bytes(observed) == canonical_bytes(profile), "target profile differs from control")


def _working_directory(destination):
    destination = Path(destination).absolute()
    _need(not destination.exists(), "fresh destination required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
    work.mkdir()
    # On failure retain only this private partial for diagnosis; never publish it.
    return work


def _publish(work, destination):
    _need(not Path(destination).exists(), "fresh destination required")
    work.rename(destination)


def _replace_working_copy(work, member, source, old_hash, new_hash):
    """Internal only: replace an explicitly bound member in OUR unpublished copy."""
    target = _path(work, member)
    _need(target.is_file() and file_sha256(target) == old_hash, "working-copy preimage differs")
    temporary = target.with_name(target.name + ".replace-" + uuid.uuid4().hex)
    with source.open("rb") as src, temporary.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    _need(file_sha256(temporary) == new_hash, "replacement hash differs")
    os.replace(temporary, target)


def prepare(spec, input_root, destination, *, replacements=None):
    """Copy exact inputs, optionally replacing authenticated extracted members.

    replacements maps a final member to {source: relative path, previous_sha256}.
    Both sources are beneath input_root. Nothing mutates the original, and no
    symlinks/hardlinks are used. Large immutable corpus mounts belong to family
    validators; bind their inventory in inputs, not an unverified directory.
    An existing exact stage is reused only after checking every byte and its
    complete inventory. A changed spec or damaged stage is never overwritten.
    """
    replacements = {} if replacements is None else replacements
    _need(type(replacements) is dict and set(replacements) <= set(spec.inventory), "unknown replacement")
    if Path(destination).exists() or _linked(Path(destination)):
        return validate_stage(spec, destination)
    work = _working_directory(destination)
    for member, digest in spec.inventory.items():
        source = _path(input_root, member)
        preimage = digest
        if member in replacements:
            record = replacements[member]
            _keys(record, {"source", "previous_sha256"}, "replacement")
            _hash(record["previous_sha256"], "replacement preimage")
            preimage = record["previous_sha256"]
        _need(source.is_file() and file_sha256(source) == preimage, "staging source differs: " + member)
        target = _path(work, member)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        _need(file_sha256(target) == preimage, "staging copy changed")
        if member in replacements:
            _replace_working_copy(work, member, _path(input_root, replacements[member]["source"]),
                                  preimage, digest)
    create_new(work / "arm-spec.json", spec.encoded)
    receipt = {"schema": "arm-stage/v1", "spec_sha256": spec.sha256,
               "inputs": spec.inventory, "status": "STAGED_NO_EXECUTION"}
    create_new(work / "stage-receipt.json", canonical_bytes(receipt))
    validate_stage(spec, work)
    _publish(work, destination)
    return receipt


def validate_stage(spec, stage):
    _need(_files(stage) == set(spec.inventory) | STAGE_METADATA, "stage inventory differs")
    validate_inputs(spec, stage)
    _need((Path(stage) / "arm-spec.json").read_bytes() == spec.encoded, "staged spec differs")
    receipt = parse_json((Path(stage) / "stage-receipt.json").read_bytes())
    _need(receipt == {"schema": "arm-stage/v1", "spec_sha256": spec.sha256,
                      "inputs": spec.inventory, "status": "STAGED_NO_EXECUTION"}, "stage receipt differs")
    return receipt


def check_target_observation(spec, observation):
    """Caller must collect this on the target now; no registration-host fallback."""
    _keys(observation, {"hostname", "runtime", "backend_policy"}, "target observation")
    check_environment(spec.document["target_environment"], hostname=observation["hostname"],
                      runtime=observation["runtime"])
    _need(canonical_bytes(observation["backend_policy"]) ==
          canonical_bytes(spec.document["config"]["backend_policy"]), "target backend policy differs")


def verify_evidence(spec, note, root):
    """Bind command evidence and actual artifact bytes, including both smokes.

    Logs are strict JSON at evidence/<log_sha256>.json, not a PASS substring.
    A trusted runner/reviewer is responsible for producing truthful records.
    """
    validate_verification_note(note, **spec.verification_args)
    records = [(check, None) for check in note["checks"]]
    records += [(smoke, kind) for kind, smoke in note["smokes"].items() if smoke is not None]
    for record, kind in records:
        member = "evidence/" + record["log_sha256"] + ".json"
        raw = _path(root, member).read_bytes()
        _need(hashlib.sha256(raw).hexdigest() == record["log_sha256"], "evidence log hash differs")
        log = parse_json(raw)
        _keys(log, {"schema", "command", "exit_code", "artifacts", "smoke"}, "command evidence")
        _need(log["schema"] == EVIDENCE_SCHEMA and type(log["exit_code"]) is int and
              log["command"] == record["command"] and log["exit_code"] == record["exit_code"],
              "command evidence differs")
        _need(type(log["artifacts"]) is dict, "evidence artifacts required")
        for role, artifact in log["artifacts"].items():
            _text(role, "artifact role")
            _keys(artifact, {"path", "sha256"}, "evidence artifact")
            _verify_files(root, {artifact["path"]: artifact["sha256"]})
        if kind is None:
            _need(log["smoke"] is None, "check evidence contains smoke")
        else:
            _need({"checkpoint", "readout"} <= set(log["artifacts"]), "smoke artifacts incomplete")
            smoke_keys = {"hostname", "device", "environment_sha256", "bindings_sha256", "results"}
            _keys(log["smoke"], smoke_keys | {"observation"}, "smoke evidence")
            _need(canonical_bytes({k: log["smoke"][k] for k in smoke_keys}) ==
                  canonical_bytes({k: record[k] for k in smoke_keys}), "smoke evidence differs")
            observation = log["smoke"]["observation"]
            if kind == "target":
                check_target_observation(spec, observation)
            else:
                _need(type(observation) is dict and observation.get("hostname") == record["hostname"]
                      and content_sha256(observation) == record["environment_sha256"],
                      "CPU observation differs")
    return content_sha256(note)


def admission_dry_run(spec, stage, note, evidence_root):
    """CPU integrity preflight only; never launch permission or a GPU probe."""
    validate_stage(spec, stage)
    note_hash = verify_evidence(spec, note, evidence_root)
    status = "EVIDENCE_READY_TARGET_CHECK_REQUIRED" if note["verdict"] == "PASS" else "HOLD"
    return {"status": status, "launch_authorized": False, "spec_sha256": spec.sha256,
            "verification_sha256": note_hash, "blockers": note["blockers"]}


def require_execution_evidence(spec, stage, note, evidence_root, *, fresh_observation):
    """Integrity gate for future family runners, not a scheduler/launcher.

    The caller still owns real resource admission and scientific contract checks.
    """
    validate_stage(spec, stage)
    note_hash = require_verification_pass(note, **spec.verification_args)
    verify_evidence(spec, note, evidence_root)
    check_target_observation(spec, fresh_observation)
    return note_hash


def artifact_receipt(spec, output_root, *, verification_sha256, exit_code, elapsed_seconds, metrics):
    """Hash complete outputs. This is neither a model validator nor a screen."""
    _hash(verification_sha256, "receipt verification")
    _need(type(exit_code) is int and exit_code == 0, "process did not complete successfully")
    _need(type(elapsed_seconds) in (int, float) and math.isfinite(elapsed_seconds) and elapsed_seconds >= 0,
          "elapsed measurement must be finite and nonnegative")
    _need(type(metrics) is dict and bool(metrics), "receipt metrics required")
    for name, value in metrics.items():
        _text(name, "metric")
        _need(type(value) in (int, float) and math.isfinite(value), "metric must be finite")
    _need(_files(output_root) == set(spec.document["outputs"]), "output inventory incomplete or extra")
    hashes = {name: file_sha256(_path(output_root, name)) for name in spec.document["outputs"]}
    return {"schema": RECEIPT_SCHEMA, "tool_version": VERSION, "scope": spec.scope,
            "spec_sha256": spec.sha256, "verification_sha256": verification_sha256,
            "status": "ARTIFACTS_VERIFIED", "scientific_status": "UNASSESSED",
            "exit_code": exit_code, "elapsed_seconds": elapsed_seconds, "metrics": metrics,
            "outputs": hashes}


def verify_artifact_receipt(spec, output_root, receipt, *, expected_receipt_sha256):
    _hash(expected_receipt_sha256, "expected receipt")
    _need(content_sha256(receipt) == expected_receipt_sha256, "receipt hash differs")
    _keys(receipt, {"schema", "tool_version", "scope", "spec_sha256", "verification_sha256",
                    "status", "scientific_status", "exit_code", "elapsed_seconds", "metrics", "outputs"},
          "artifact receipt")
    _need(type(receipt["tool_version"]) is str and
          re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", receipt["tool_version"]) is not None,
          "receipt tool version invalid")
    actual = artifact_receipt(spec, output_root, verification_sha256=receipt["verification_sha256"],
                              exit_code=receipt["exit_code"], elapsed_seconds=receipt["elapsed_seconds"],
                              metrics=receipt["metrics"])
    # The schema controls interpretation; producer version is bound provenance.
    # A later compatible verifier must still collect an already emitted receipt.
    actual["tool_version"] = receipt["tool_version"]
    _need(canonical_bytes(actual) == canonical_bytes(receipt), "artifact receipt differs")
    return actual


class _HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        return block


def export_artifacts(spec, output_root, receipt, archive, *, expected_receipt_sha256):
    verify_artifact_receipt(spec, output_root, receipt, expected_receipt_sha256=expected_receipt_sha256)
    # Exclusive tar creation; a failure may leave an unpublished/invalid archive,
    # which cannot pass collection against the trusted receipt identity.
    with tarfile.open(archive, "x") as target:
        raw = canonical_bytes(receipt)
        info = tarfile.TarInfo("receipt.json")
        info.size = len(raw)
        target.addfile(info, io.BytesIO(raw))
        for member in sorted(receipt["outputs"]):
            # Explicit regular members also handle hardlinked local outputs.
            # Hash the bytes actually archived, detecting a concurrent writer.
            with _path(output_root, member).open("rb") as stream:
                info = tarfile.TarInfo("outputs/" + member)
                info.size = os.fstat(stream.fileno()).st_size
                reader = _HashingReader(stream)
                target.addfile(info, reader)
                _need(reader.digest.hexdigest() == receipt["outputs"][member], "export output changed")
    return file_sha256(archive)


def collect_artifacts(spec, archive, destination, *, expected_receipt_sha256):
    """Reject aliases, links, duplicates, omissions and drift before publishing."""
    expected = {"receipt.json"} | {"outputs/" + name for name in spec.document["outputs"]}
    archive_hash = file_sha256(archive)
    with tarfile.open(archive, "r:*") as source:
        members = source.getmembers()
        for member in members:
            safe_member(member.name)
            _need(member.isfile() and not member.sparse, "nonregular archive member")
        _need(len(members) == len(expected) and {m.name for m in members} == expected,
              "archive inventory differs")
        work = _working_directory(destination)
        for member in members:
            path = _path(work, member.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as src, path.open("xb") as dst:
                shutil.copyfileobj(src, dst)
    receipt = parse_json((work / "receipt.json").read_bytes())
    verify_artifact_receipt(spec, work / "outputs", receipt, expected_receipt_sha256=expected_receipt_sha256)
    _need(file_sha256(archive) == archive_hash, "archive changed during collection")
    result = {"schema": "arm-collection/v1", "status": "COLLECTED_ARTIFACTS_VERIFIED",
              "scientific_status": "UNASSESSED", "spec_sha256": spec.sha256,
              "receipt_sha256": expected_receipt_sha256, "archive_sha256": archive_hash}
    create_new(work / "collection.json", canonical_bytes(result))
    _publish(work, destination)
    return result


def prediction_csv_bytes(ids, values, targets, *, id_column="study_id"):
    """Preserve the historical float32 -> pandas CSV -> float64 boundary."""
    import numpy as np
    import pandas as pd

    ids, targets = list(ids), list(targets)
    _need(bool(ids) and all(type(i) is str and i for i in ids) and len(set(ids)) == len(ids),
          "prediction IDs invalid")
    _text(id_column, "prediction ID column")
    _need(bool(targets) and all(type(t) is str and t.strip() for t in targets) and
          len(set(targets)) == len(targets) and id_column not in targets,
          "prediction columns invalid")
    values = np.asarray(values, dtype=np.float32)
    _need(values.shape == (len(ids), len(targets)) and np.isfinite(values).all(), "prediction values invalid")
    frame = pd.DataFrame(values, columns=targets)
    frame.insert(0, id_column, ids)
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def load_prediction_csv(path, ids, targets, *, id_column="study_id"):
    """Align exact study/target sets, retain pandas float64 readout, never print rows."""
    import csv
    import numpy as np
    import pandas as pd

    ids, targets = list(ids), list(targets)
    _text(id_column, "prediction ID column")
    _need(bool(targets) and all(type(t) is str and t.strip() for t in targets),
          "prediction columns invalid")
    _need(all(type(i) is str and i for i in ids), "prediction IDs invalid")
    with Path(path).open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream), [])
    _need(header == [id_column] + targets and len(set(header)) == len(header), "prediction header differs")
    frame = pd.read_csv(path, dtype={id_column: str})
    _need(bool(ids) and len(set(ids)) == len(ids) and len(frame) == len(ids) and
          frame[id_column].notna().all() and frame[id_column].is_unique and
          set(frame[id_column]) == set(ids), "prediction roster differs")
    values = frame.set_index(id_column).loc[ids, targets].to_numpy(dtype=np.float64)
    _need(np.isfinite(values).all(), "prediction values nonfinite")
    return values
