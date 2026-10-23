"""Strict pre-push gate v1.3.0 (CPU-only, never executes notebook code).

Changelog 1.3.0: additive pinned public-R/R-KD and HB151/ConvNeXt-S contracts.
Backup: .local/backups/strict-variant-gate-v1.2.0/.
Changelog 1.2.0: pinned EXP143v2 parent, HB151 external archive and exact 5-to-6 FAST binding.
Changelog 1.1.0: pinned reviewed FAST cell/guard, dispatcher-derived identity,
and create-only bind helper for registration, manifest and review bindings.
Changelog 1.0.0: shared exhaustive archive/cell/metadata diff, manifest and
independent-lane review binding, immutable hash-bound admission receipt.
See tools/strict_variant_gate.md for the intentionally strict allowlist schema.
"""
from __future__ import annotations

import argparse
import ast
import base64
import difflib
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
import zipfile

VERSION = "1.2.0"
TOOL_VERSION = "1.3.0"  # Preserve the legacy receipt protocol byte-for-byte.
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import strict_variant_gate_hb151 as hb151
from tools import strict_variant_gate_standalone as standalone
ROOT = Path(__file__).resolve().parents[1]
HEX = r"[0-9a-f]{64}"
PROTECTED = ("docker_image", "dataset_sources", "machine_shape", "enable_internet")
MAX_BYTES = 64 * 1024 * 1024
RELEASES = "artifacts/experiments/EXP169_RELEASE_C85R7/"
PRESUB = ".local/scratch/slot-b/presub/"
# Explicit migration anchors for the two reviews named by B. No arbitrary
# historical prose review can be promoted to a manifest-bound approval.
HISTORICAL = {
    "v1.1.18-own25q3": {
        "kernel_files": {"kernel-metadata.json": "6410ce0c932caf2247bc02e484a1dc364dd4b415d8cf9a59a8f72963688ac7a1", "release.ipynb": "98668e17d4419249ccce63641b17b90e5ec9bb3612f72ffb875ef3b33d3f5855"},
        "review": PRESUB + "own25q3-review-out.md", "review_sha256": "ba50d675862d5f3919fb88792416b67b43987563028b00d108cc54cc72509389",
        "builder_receipt_sha256": "5f627b175be9cd0b8ebc00976d8d00ad03a511ae699616fc8756384fa9b53679",
    },
    "v1.1.19-p65q2": {
        "kernel_files": {"kernel-metadata.json": "0820cbcc3eae9be9d3e342520ebd60a1347f1fe785b817bd5a81023b261827bb", "release.ipynb": "da4af9cde08b33cb9f114ce6c11425de8a857aff0ed1ddb413764b363a49b6b6"},
        "review": PRESUB + "p65q2-review-out.md", "review_sha256": "68be6d355c87cc777b1e0062b6f9404f5f7e8ed2595c219ba4296cdfe104e609",
        "builder_receipt_sha256": "4bb5fe41c6ea3cc878f2460215946a0c6e04d28b9e766204807bacf0ec9096fb",
    },
}
FAST_REFERENCE = {
    "stage": RELEASES + "v1.1.18-own25q3",
    "review": HISTORICAL["v1.1.18-own25q3"]["review"],
    "review_sha256": HISTORICAL["v1.1.18-own25q3"]["review_sha256"],
    "cell0_sha256": "14e0f93fb67c435c49966b891955e7b2bf721b012dd72c64c3482d068a212ef2",
    "cell0_object_sha256": "8c206ac888ba4a9be5f8244c473b03226c17652d04e733c4caa933467df53ddd",
}


class GateError(ValueError):
    pass


def need(ok, message):
    if not ok:
        raise GateError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse(data):
    return json.loads(data, object_pairs_hook=unique_object)


def safe_name(name):
    need(isinstance(name, str) and name and not any(x in name for x in ("\\", ":", "\x00")), "unsafe relative path")
    p = PurePosixPath(name)
    need(not p.is_absolute() and all(x not in ("", ".", "..") for x in name.split("/")), "unsafe relative path")
    return name


def local_path(path, root):
    path = Path(path)
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    need(path.is_relative_to(root.resolve()), "input outside project")
    parts = [x.lower() for x in path.relative_to(root.resolve()).parts]
    need(not any(x in parts for x in (".kaggle", ".codex", ".ssh", "api-keys", ".relabel")) and parts[:2] != [".local", "relabel"], "forbidden input path")
    return path


class Inputs:
    """Snapshot every input; detect concurrent changes before granting admission."""
    def __init__(self, root):
        self.root = root.resolve()
        self.data = {}
        self.folders = {}
        self.digests = {}

    def read(self, path):
        path = local_path(path, self.root)
        if path not in self.data:
            need(path.stat().st_size <= MAX_BYTES, "oversize input: " + str(path))
            self.data[path] = path.read_bytes()
        return self.data[path]

    def digest(self, path):
        path = local_path(path, self.root)
        if path not in self.digests:
            with path.open("rb") as stream:
                self.digests[path] = hashlib.file_digest(stream, "sha256").hexdigest()
        return self.digests[path]

    def json(self, path):
        return parse(self.read(path))

    def hashes(self):
        return {**{p.relative_to(self.root).as_posix(): h for p, h in sorted(self.digests.items())}, **{p.relative_to(self.root).as_posix(): sha(b) for p, b in sorted(self.data.items())}}

    def recheck(self):
        for path, digest in self.digests.items():
            with path.open("rb") as stream:
                need(hashlib.file_digest(stream, "sha256").hexdigest() == digest, "input changed during gate: " + str(path))
        for path, data in self.data.items():
            need(path.read_bytes() == data, "input changed during gate: " + str(path))
        for folder, paths in self.folders.items():
            need(sorted(p for p in folder.rglob("*") if p.is_file()) == paths, "kernel file set changed during gate")


def manifest(stage, inputs, required):
    candidates = [stage / n for n in ("SHA256SUMS", "SHA256SUMS.txt") if (stage / n).is_file()]
    need(len(candidates) == 1, "stage needs exactly one SHA256SUMS or SHA256SUMS.txt")
    path = candidates[0]
    entries = {}
    for line in inputs.read(path).decode("utf-8-sig").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        match = re.fullmatch(r"(" + HEX + r") [ *](.+)", line)
        need(match is not None, "malformed SHA256SUMS line")
        digest, name = match.groups()
        safe_name(name)
        # Existing C85 manifests use project-relative paths; small stages may
        # use stage-relative ones. Ambiguous resolution is never accepted.
        choices = {local_path(p, inputs.root) for p in (stage / name, inputs.root / name) if p.is_file()}
        need(len(choices) == 1, "missing/ambiguous manifest path: " + name)
        target = choices.pop()
        need(target not in entries and target != path, "duplicate/self manifest entry")
        entries[target] = digest
    for target in required:
        need(target in entries, "SHA256SUMS does not cover " + target.name)
        need(sha(inputs.read(target)) == entries[target], "SHA256SUMS mismatch: " + target.name)
    # Ancillary logs can be historical; do not read unrelated datasets/labels.
    # Required stage inputs are all authenticated, and the full manifest is bound.
    return path, sha(inputs.read(path))


def unpack_notebook(raw):
    nb = parse(raw)
    need(isinstance(nb.get("cells"), list), "notebook cells missing")
    inventory = {"notebook/top.json": encoded({k: v for k, v in nb.items() if k != "cells"})}
    bindings = {"BLOB": [], "PINS": []}
    sources = {}
    for i, cell in enumerate(nb["cells"]):
        source = cell["source"]
        need(isinstance(source, str) or (isinstance(source, list) and all(isinstance(s, str) for s in source)), "invalid cell source")
        source = source if isinstance(source, str) else "".join(source)
        sources[i] = source
        inventory[f"notebook/cells/{i}/fields.json"] = encoded({k: v for k, v in cell.items() if k != "source"})
        if cell["cell_type"] == "code":
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in bindings:
                            need(len(node.targets) == 1, "chained payload assignment")
                            bindings[target.id].append((i, node.value))
    need(all(len(v) == 1 for v in bindings.values()), "exactly one literal BLOB and PINS required across ALL cells")
    values = {key: ast.literal_eval(nodes[0][1]) for key, nodes in bindings.items()}
    need(isinstance(values["BLOB"], str) and isinstance(values["PINS"], dict), "invalid BLOB/PINS types")
    blob = base64.b64decode(values["BLOB"], validate=True)
    need(len(blob) <= MAX_BYTES, "oversize archive")
    files = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        members = archive.infolist()
        need(sum(m.file_size for m in members) <= MAX_BYTES, "oversize unpacked archive")
        seen = set()
        for info in members:
            name = safe_name(info.filename)
            need(name.casefold() not in seen, "duplicate/case-colliding archive member")
            seen.add(name.casefold())
            need(not info.is_dir() and (info.external_attr >> 16) & 0o170000 != 0o120000, "directory/symlink archive member")
            files[name] = archive.read(info)
    need(files and {k: sha(v) for k, v in files.items()} == values["PINS"], "embedded PINS mismatch")
    # Normalize only the validated payload literal expressions. Assignment,
    # indentation, bootstrap and archive-hash assertion stay in the line diff.
    for i, source in sources.items():
        data = source.encode()
        lines = data.splitlines(keepends=True)
        replacements = []
        for key, nodes in bindings.items():
            cell_i, node = nodes[0]
            if cell_i == i:
                start = sum(map(len, lines[:node.lineno - 1])) + node.col_offset
                end = sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset
                replacements.append((start, end, repr("<verified-" + key + ">").encode()))
        for start, end, replacement in sorted(replacements, reverse=True):
            data = data[:start] + replacement + data[end:]
        inventory[f"notebook/cells/{i}/source.txt"] = data
    inventory.update({"embedded/" + k: v for k, v in files.items()})
    return inventory, {"archive_sha256": sha(blob), "embedded": {k: sha(v) for k, v in files.items()}}


def kernel(stage, inputs):
    folder = stage / ("own" if stage == inputs.root / hb151.PARENT else "kernel")
    need(folder.is_dir(), "kernel directory missing")
    paths = sorted(p for p in folder.rglob("*") if p.is_file())
    inputs.folders[folder] = paths
    need(all(not p.is_symlink() and p.resolve().is_relative_to(folder.resolve()) for p in paths), "linked kernel file")
    raw = {p.relative_to(folder).as_posix(): inputs.read(p) for p in paths}
    if stage == inputs.root / hb151.PARENT:
        # Original upload folder contains a non-executed description alongside
        # the two kernel files. Authenticate it before making the binding copy.
        need(sha(raw.pop("DESCRIPTION.md")) == "6c8bce2d0aec560ec8ade5686c5fd120701c0f5cecd22d5cc74d09bb9626a26f", "parent description changed")
    meta = parse(raw["kernel-metadata.json"])
    if meta.get("id") in standalone.IDS:
        return standalone.kernel(stage, inputs, sys.modules[__name__])
    if meta.get("id") in (hb151.PARENT_ID, hb151.CHILD_ID):
        return hb151.kernel(stage, inputs, raw, paths, sys.modules[__name__])
    code_file = safe_name(meta["code_file"])
    need(code_file.endswith(".ipynb"), "only notebook C85-R7 kernels supported")
    inventory, details = unpack_notebook(raw[code_file])
    inventory.update({"kernel/" + k: v for k, v in raw.items() if k != code_file})
    details["kernel_files"] = {k: sha(v) for k, v in raw.items()}
    return meta, inventory, details, raw, paths


def compare(before, after, rules):
    errors, full_diff = [], []
    if not isinstance(rules, dict):
        errors.append("allowlist files must be an object")
        rules = {}
    valid_rules = {}
    for name, rule in rules.items():
        try:
            need(name in before and name in after, "allowlist references absent file: " + name)
            need(set(rule) == {"before_sha256", "after_sha256", "removed", "added"}, "invalid allowlist rule: " + name)
            need(all(re.fullmatch(HEX, rule[k]) for k in ("before_sha256", "after_sha256")), "invalid allowlist hash")
            for key in ("removed", "added"):
                need(isinstance(rule[key], list) and all(isinstance(p, str) for p in rule[key]), "invalid line patterns")
                for pattern in rule[key]:
                    re.compile(pattern)
            need(sha(before[name]) == rule["before_sha256"] and sha(after[name]) == rule["after_sha256"], "allowlist hash mismatch: " + name)
            valid_rules[name] = rule
        except (ValueError, TypeError, KeyError, re.error) as exc:
            errors.append(str(exc))
    for name in sorted(before.keys() | after.keys()):
        a, b = before.get(name), after.get(name)
        if a == b:
            continue
        if a is None or b is None:
            errors.append("file/cell added, deleted or renamed: " + name)
        rule = valid_rules.get(name)
        if rule is None:
            errors.append("unallowed file: " + name)
        try:
            old, new = (a or b"").decode().splitlines(keepends=True), (b or b"").decode().splitlines(keepends=True)
        except UnicodeDecodeError:
            errors.append("binary change forbidden: " + name)
            full_diff.append(f"BINARY {name}: {sha(a or b'')} -> {sha(b or b'')}\n")
            continue
        full_diff.extend(difflib.unified_diff(old, new, fromfile="parent/" + name, tofile="stage/" + name))
        if rule:
            matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
            for op, i, j, k, l in matcher.get_opcodes():
                if op == "equal":
                    continue
                for side, lines in (("removed", old[i:j]), ("added", new[k:l])):
                    for line in lines:
                        if not any(re.fullmatch(p, line.rstrip("\r\n")) for p in rule[side]):
                            errors.append("unallowed " + side + " line: " + name)
    return errors, "".join(full_diff)


def verdict(data):
    text = "\n".join(data.decode("utf-8-sig").splitlines())
    need(text.rstrip().endswith("\nVERDICT: PASS") or text.rstrip() == "VERDICT: PASS", "review must end with VERDICT: PASS")
    need(re.findall(r"^VERDICT:.*$", text, re.M) == ["VERDICT: PASS"], "ambiguous review verdict")
    return text


def dispatch(path, inputs):
    """Resolve identity through the dispatcher's archived run, not allow labels.

    The project dispatcher archive is the trust boundary (not a signature or
    protection against an adversary with write access to that archive).
    """
    record = inputs.json(path)
    need(record["schema"] == "dispatch-terminal/v1" and record["state"] == "SUCCESS" and type(record["exit_code"]) is int and record["exit_code"] == 0, "unsuccessful dispatch receipt")
    lane, run_id = record["lane"], record["run_id"]
    need(re.fullmatch(r"[A-Za-z0-9_-]+", lane) and re.fullmatch(r"[0-9a-f]{32}", run_id), "invalid dispatch identity")
    archive = inputs.root / ".local/astra-improve/lanes/runs" / lane / run_id
    need(inputs.json(archive / "receipt.json") == record, "dispatch archive receipt mismatch")
    report = inputs.read(record["out"])
    need(sha(report) == record["report_sha256"] and report == inputs.read(archive / "report.md"), "dispatch report mismatch")
    need(sha(inputs.read(archive / "prompt.txt")) == record["prompt_sha256"], "dispatch prompt mismatch")
    return lane, report, local_path(record["out"], inputs.root)


def source_binding(allow, stage_info, inputs):
    binding = allow["review_binding"]
    need(set(binding) == {"source_stage", "review", "builder_receipt", "builder_receipt_sha256"}, "invalid review binding")
    source = local_path(binding["source_stage"], inputs.root)
    original = kernel(source, inputs)
    need(original[2]["kernel_files"] == stage_info[2]["kernel_files"], "bound source kernel differs from stage")
    receipt = inputs.read(binding["builder_receipt"])
    need(sha(receipt) == binding["builder_receipt_sha256"], "builder receipt hash mismatch")
    lane, report, _ = dispatch(binding["builder_receipt"], inputs)
    anchor = HISTORICAL.get(source.name) if source == local_path(RELEASES + source.name, inputs.root) else None
    if anchor:
        need(original[2]["kernel_files"] == anchor["kernel_files"] and sha(receipt) == anchor["builder_receipt_sha256"], "historical stage/builder binding mismatch")
    else:
        digest = sha(encoded(original[2]["kernel_files"]))
        need(re.findall(r"^Kernel-SHA256: (" + HEX + r")\s*$", report.decode("utf-8-sig"), re.M) == [digest], "builder report must bind Kernel-SHA256")
    return lane, anchor


def review_check(path, allow, manifest_sha, inputs, stage_info, allow_path):
    lane, anchor = source_binding(allow, stage_info, inputs)
    path = local_path(path, inputs.root)
    need(path == local_path(allow["review_binding"]["review"], inputs.root), "review binding path mismatch")
    scratch = inputs.root / ".local/scratch" / lane
    need(not path.is_relative_to(scratch), "review is under builder scratch; different-lane review required")
    text = verdict(inputs.read(path))
    if anchor:
        need(path == local_path(anchor["review"], inputs.root) and sha(inputs.read(path)) == anchor["review_sha256"], "historical review hash/path mismatch")
        # These exact two review/stage pairings are the user-authorized migration.
        need(lane != "slot-b", "historical review is not independent")
        return "slot-b"
    reviewer, _, report_path = dispatch(str(path) + ".receipt.json", inputs)
    need(report_path == path and reviewer != lane, "different-lane dispatched review required")
    need(re.findall(r"^Reviewer-Lane: ([A-Za-z0-9_-]+)\s*$", text, re.M) == [reviewer], "review lane differs from dispatcher")
    need(re.findall(r"^SHA256SUMS(?:\.txt)?: (" + HEX + r")\s*$", text, re.M) == [manifest_sha], "review does not name current SHA256SUMS hash")
    need(re.findall(r"^Allowlist-SHA256: (" + HEX + r")\s*$", text, re.M) == [sha(inputs.read(allow_path))], "review does not name current allowlist hash")
    return reviewer


def fast_reference(policy, inputs):
    need(policy == FAST_REFERENCE, "FAST reference must be the pinned reviewed own25q3 cell")
    reference = kernel(local_path(policy["stage"], inputs.root), inputs)
    need(reference[2]["kernel_files"] == HISTORICAL["v1.1.18-own25q3"]["kernel_files"], "FAST reference kernel changed")
    need(sha(inputs.read(policy["review"])) == policy["review_sha256"], "FAST reference review changed")
    verdict(inputs.read(policy["review"]))


def fast_inventory(old, new, policy, inputs):
    if "external_policy" in new[2]:
        return hb151.fast_inventory(old, new, policy, inputs, sys.modules[__name__])
    fast_reference(policy, inputs)
    before = parse(old[3][old[0]["code_file"]])
    after = parse(new[3][new[0]["code_file"]])
    need(len(before["cells"]) == 2 and len(after["cells"]) == 3, "FAST requires exactly parent two cells plus cell 0")
    cell = after["cells"][0]
    need(sha(encoded(cell)) == policy["cell0_object_sha256"] and sha("".join(cell["source"]).encode()) == policy["cell0_sha256"], "FAST cell 0 hash mismatch")
    need(before["cells"][0]["cell_type"] == "markdown" and before["cells"][1]["cell_type"] == "code", "unsupported FAST parent layout")
    need(after["cells"][1] == before["cells"][0], "FAST markdown changed")
    after["cells"] = after["cells"][1:]
    source = "".join(after["cells"][1]["source"])
    lines = source.splitlines(keepends=True)
    need(lines[0] == "if not FAST:\n" and all(line.startswith("    ") for line in lines[1:]), "FAST guard must be exact if not FAST plus four-space wrapper")
    body = "".join(line[4:] for line in lines[1:])
    need(not any(isinstance(n, ast.Name) and n.id == "FAST" for n in ast.walk(ast.parse(body))), "FAST may only occur in the pinned cell and guard")
    after["cells"][1]["source"] = body.splitlines(keepends=True)
    inventory, _ = unpack_notebook(encoded(after))
    inventory.update({k: v for k, v in new[1].items() if k.startswith("kernel/")})
    for name, content in old[1].items():
        if name.startswith("notebook/"):
            candidate = inventory.get(name)
            if name == "notebook/cells/1/source.txt":
                # Only the already verified archive hash may vary in the run cell.
                a = content.replace(old[2]["archive_sha256"].encode(), b"<archive>")
                b = candidate.replace(new[2]["archive_sha256"].encode(), b"<archive>")
                need(a == b, "FAST run cell has changes beyond payload/guard")
            else:
                need(content == candidate, "FAST notebook fields/top changed: " + name)
    return inventory


def evaluate(stage, parent, allow_path, review, root=ROOT):
    if standalone.recognizes(local_path(stage, root), sys.modules[__name__]):
        return standalone.evaluate(stage, parent, allow_path, review, root, sys.modules[__name__])
    inputs = Inputs(root)
    stage, parent = local_path(stage, root), local_path(parent, root)
    errors = []
    def check(action):
        try:
            return action()
        except (ValueError, OSError, KeyError, TypeError, SyntaxError, zipfile.BadZipFile) as exc:
            errors.append(str(exc))
            return None
    allow = inputs.json(allow_path)
    need(set(allow) - {"fast_path", "external_archive"} == {"version", "review_binding", "parent_sha256sums", "stage_sha256sums", "files"} and allow["version"] in {"1.1.0", VERSION}, "invalid allowlist schema/version")
    old = check(lambda: kernel(parent, inputs))
    new = check(lambda: kernel(stage, inputs))
    stage_manifest = check(lambda: manifest(stage, inputs, ([] if not new else new[4]) + [stage / "registration.json"] + ([stage / "full-diff.txt"] if new and new[2].get("external_policy") else [])))
    parent_manifest = check(lambda: manifest(parent, inputs, ([] if not old else old[4]) + [parent / "registration.json"]))
    reviewer = check(lambda: review_check(review, allow, stage_manifest[1] if stage_manifest else "INVALID", inputs, new, allow_path)) if new else None
    if stage_manifest:
        check(lambda: need(stage_manifest[1] == allow["stage_sha256sums"], "allowlist stage manifest mismatch"))
    if parent_manifest:
        check(lambda: need(parent_manifest[1] == allow["parent_sha256sums"], "allowlist parent manifest mismatch"))
    diff = ""
    if new and old:
        meta, before_meta = new[0], old[0]
        kernel_id = meta.get("id", "")
        check(lambda: need(re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", kernel_id), "invalid kernel id"))
        check(lambda: need("/" not in meta.get("title", "/") and meta.get("title") == kernel_id.split("/")[-1], "title must equal id slug without '/'"))
        check(lambda: need(inputs.json(stage / "registration.json")["kernel_id"] == kernel_id, "id differs from stage registration kernel_id"))
        external = new[2].get("external_policy")
        if external or "external_archive" in allow:
            check(lambda: need(allow["version"] == VERSION and external and allow.get("external_archive") == external and "fast_path" in allow, "external archive policy missing/mismatched"))
            reg = inputs.json(stage / "registration.json")
            check(lambda: need(local_path(reg["parent"], inputs.root) == parent and reg["parent_sha256sums"] == allow["parent_sha256sums"], "registered external parent mismatch"))
        for key in PROTECTED:
            if key == "dataset_sources" and external:
                continue
            check(lambda key=key: need(key in meta and key in before_meta and encoded(meta[key]) == encoded(before_meta[key]), "protected metadata differs: " + key))
        check(lambda: need(type(meta.get("enable_internet")) is bool and type(before_meta.get("enable_internet")) is bool, "enable_internet must be a JSON boolean"))
        check(lambda: need(set(new[3]) == set(old[3]), "kernel file added/deleted/renamed"))
        comparison = check(lambda: fast_inventory(old, new, allow["fast_path"], inputs)) if "fast_path" in allow else new[1]
        result = check(lambda: compare(old[1], comparison, allow["files"])) if comparison else None
        if result:
            errors.extend(result[0])
            diff = result[1]
        # Full raw notebook diff complements the decoded/normalized allowlist diff.
        for name in sorted(old[3].keys() | new[3].keys()):
            if name.endswith(".ipynb"):
                diff += "".join(difflib.unified_diff(old[3].get(name, b"").decode().splitlines(True), new[3].get(name, b"").decode().splitlines(True), fromfile="raw-parent/" + name, tofile="raw-stage/" + name))
        if external:
            check(lambda: need(inputs.read(stage / "full-diff.txt") == diff.encode(), "external full diff differs from recomputed inventory"))
    inputs.read(Path(__file__))
    check(inputs.recheck)
    # Rescan file sets too: a new unmanifested kernel file cannot race admission.
    for directory, snapshot in ((parent, old), (stage, new)):
        if snapshot:
            check(lambda directory=directory, snapshot=snapshot: need(sorted(p for p in (directory / "kernel").rglob("*") if p.is_file()) == snapshot[4], "kernel file set changed during gate"))
    return {
        "fixture_only": bool(new and new[2].get("fixture_only")),
        "gate_version": VERSION, "status": "FAIL" if errors else "PASS", "errors": sorted(set(errors)),
        "stage": str(stage), "parent": str(parent), "kernel_id": new[0].get("id") if new else None,
        "reviewer_lane": reviewer, "input_sha256": inputs.hashes(),
        "parent_inventory": old[2] if old else None, "stage_inventory": new[2] if new else None,
        "comparison_sha256": {"parent": {k: sha(v) for k, v in old[1].items()} if old else {}, "stage": {k: sha(v) for k, v in new[1].items()} if new else {}},
        "diff_sha256": sha(diff.encode()), "full_diff": diff,
    }


def write_receipt(path, result):
    """Atomic create-once receipt. Identical reruns are allowed, never overwrite."""
    need(not result.get("fixture_only"), "fixture-only gate cannot publish admission")
    payload = encoded(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    # Exclusive creation serializes concurrent gates for the same slug.
    with temp.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        for attempt in range(40):
            try:
                if path.exists():
                    need(path.read_bytes() == payload, "immutable gate receipt exists with different inputs; use a new version/slug")
                else:
                    os.link(temp, path)  # atomic publish, cannot overwrite a race winner
                break
            except FileExistsError:
                need(path.read_bytes() == payload, "receipt race with different inputs")
                break
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.05)
    finally:
        for attempt in range(40):
            try:
                temp.unlink()
                break
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.05)


def exact_rules(before, after):
    """Draft rules only: approval must bind these exact bytes independently."""
    need(before.keys() == after.keys(), "file/cell added, deleted or renamed")
    rules = {}
    for name, a in before.items():
        b = after[name]
        if a == b:
            continue
        old, new = a.decode().splitlines(), b.decode().splitlines()
        rule = dict(before_sha256=sha(a), after_sha256=sha(b), removed=[], added=[])
        for op, i, j, k, l in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
            if op != "equal":
                rule["removed"].extend(re.escape(s) for s in old[i:j])
                rule["added"].extend(re.escape(s) for s in new[k:l])
        rules[name] = rule
    return rules


def bind(stage, parent, output, builder_receipt, review, fast=False, root=ROOT):
    """Prepare a fresh, byte-identical kernel copy; never manufacture a PASS.

    Generic allowlists are drafts until a different dispatcher lane reviews
    their hash. The two explicitly pinned historical approvals are migrated.
    """
    if standalone.recognizes(local_path(stage, root), sys.modules[__name__]):
        return standalone.bind(stage, parent, output, builder_receipt, review, fast, root, sys.modules[__name__])
    inputs = Inputs(root)
    source, parent, output = [local_path(p, root) for p in (stage, parent, output)]
    need(not output.exists(), "bind output already exists; use a fresh versioned stage")
    need(not output.is_relative_to(source) and not output.is_relative_to(parent), "bind output must be outside input stages")
    new, old = kernel(source, inputs), kernel(parent, inputs)
    parent_copy = {}
    if parent == inputs.root / hb151.PARENT:
        parent_copy = hb151.parent_files(inputs, sys.modules[__name__])
        parent_sha = sha(parent_copy["SHA256SUMS"])
        parent = output / "parent"
    else:
        parent_sha = manifest(parent, inputs, old[4] + [parent / "registration.json"])[1]
    binding = {"source_stage": source.relative_to(inputs.root).as_posix(),
               "review": local_path(review, root).relative_to(inputs.root).as_posix(),
               "builder_receipt": local_path(builder_receipt, root).relative_to(inputs.root).as_posix(),
               "builder_receipt_sha256": sha(inputs.read(builder_receipt))}
    allow = dict(version=VERSION, review_binding=binding, parent_sha256sums=parent_sha,
                 stage_sha256sums="", files={})
    lane, anchor = source_binding(allow, new, inputs)
    if anchor:
        need(local_path(review, root) == local_path(anchor["review"], root) and sha(inputs.read(review)) == anchor["review_sha256"], "historical review hash/path mismatch")
        verdict(inputs.read(review))
    meta = new[0]
    need(re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", meta["id"]) and meta["title"] == meta["id"].split("/")[1], "title must equal id slug without '/'")
    need(set(old[3]) == set(new[3]), "kernel file added/deleted/renamed")
    external = new[2].get("external_policy")
    if external:
        need(fast, "HB151 requires pinned FAST policy")
        allow["external_archive"] = external
    for key in PROTECTED:
        if key == "dataset_sources" and external:
            continue
        need(encoded(meta[key]) == encoded(old[0][key]), "protected metadata differs: " + key)
    comparison = new[1]
    if fast:
        allow["fast_path"] = dict(FAST_REFERENCE)
        comparison = fast_inventory(old, new, allow["fast_path"], inputs)
    allow["files"] = exact_rules(old[1], comparison)
    # A binding-copy directory is not a new runtime build. Preserve the source
    # release version so downstream visible-run identity checks remain usable.
    source_registration = source / "registration.json"
    runtime_version = (inputs.json(source_registration).get("version", source.name.removeprefix("v"))
                       if source_registration.is_file() else source.name.removeprefix("v"))
    if external:
        runtime_version = "1.0.2"
    registration = {"version": runtime_version, "binding_version": VERSION, "kernel_id": meta["id"],
                    "parent": str(parent), "parent_sha256sums": parent_sha,
                    "source_stage": binding["source_stage"], "kernel_files": new[2]["kernel_files"],
                    "builder_lane": lane, "builder_receipt_sha256": binding["builder_receipt_sha256"],
                    "qualified_for_submission": False, "exposure": "EXPOSED", "clean_oof": False}
    if external:
        registration["external_source"] = str(source)
    files = {"kernel/" + name: data for name, data in new[3].items()}
    files["registration.json"] = encoded(registration)
    if external:
        errors, diff = compare(old[1], comparison, allow["files"])
        need(not errors, "bound external diff failed")
        diff += "".join(difflib.unified_diff(old[3]["release.ipynb"].decode().splitlines(True),
            new[3]["release.ipynb"].decode().splitlines(True), fromfile="raw-parent/release.ipynb", tofile="raw-stage/release.ipynb"))
        files["full-diff.txt"] = diff.encode()
    sums = "".join(sha(data) + "  " + name + "\n" for name, data in sorted(files.items())).encode()
    allow["stage_sha256sums"] = sha(sums)
    files["SHA256SUMS"] = sums
    files["allowlist.json"] = encoded(allow)
    files["review-request.md"] = (
        "Review request only; not an approval. Inspect source provenance and the full allowlist.\n"
        f"Builder-Lane (dispatcher): {lane}\n"
        "Reviewer-Lane: <independent dispatched lane>\n"
        f"SHA256SUMS: {sha(sums)}\nAllowlist-SHA256: {sha(files['allowlist.json'])}\n"
        f"Kernel-SHA256: {sha(encoded(new[2]['kernel_files']))}\nVERDICT: PENDING\n").encode()
    files.update({"parent/" + name: data for name, data in parent_copy.items()})
    inputs.recheck()
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    return {"status": "BOUND" if anchor else "REVIEW_REQUIRED", "stage": str(output),
            "parent": str(parent), "allow": str(output / "allowlist.json"), "review": str(local_path(review, root)),
            "sha256sums": sha(sums), "builder_lane": lane}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    binding = bool(argv and argv[0] == "bind")
    if binding:
        argv.pop(0)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (("stage", "parent", "output", "builder-receipt", "review") if binding else ("stage", "parent", "allow", "review")):
        parser.add_argument("--" + name, type=Path, required=True)
    if binding:
        parser.add_argument("--fast", action="store_true", help="pin the reviewed own25q3 FAST cell/guard")
    else:
        parser.add_argument("--check-only", action="store_true", help="evaluate without publishing an admission receipt")
    args = parser.parse_args(argv)
    try:
        if binding:
            print(json.dumps(bind(args.stage, args.parent, args.output, args.builder_receipt, args.review, args.fast)))
            return 0
        result = evaluate(args.stage, args.parent, args.allow, args.review)
        if result["status"] != "PASS":
            # A failed invocation never creates an admission receipt.
            print(json.dumps({k: v for k, v in result.items() if k != "full_diff"}, indent=2))
            print(result["full_diff"])
            return 1
        if args.check_only:
            print(json.dumps({k: v for k, v in result.items() if k != "full_diff"}, indent=2))
            return 0
        slug = result["kernel_id"].split("/")[1]
        receipt = ROOT / ".local/state/gates" / (slug + ".json")
        write_receipt(receipt, result)
        print(json.dumps({"status": "PASS", "receipt": str(receipt), "receipt_sha256": sha(receipt.read_bytes()), "diff_sha256": result["diff_sha256"]}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, SyntaxError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
