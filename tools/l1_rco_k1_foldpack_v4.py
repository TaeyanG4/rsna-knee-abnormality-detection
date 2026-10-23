"""H224 portable bundle v4.0.0: additive all-data refit plus unchanged fold tools.

Changelog v4.0.0: fixed Vast f2-only / local f3 schedule, local waiter, holdout
gate and new archive. Preserve earlier archives and all training registrations.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools import l1_rco_k1_all_v1 as k
from tools import l1_rco_k1_foldpack_v3 as previous
from tools import k1_fold_dataset_v1 as assets

PACKAGE = ROOT / ".local/scratch/astra-b2/h224-device-plan-v1/package-v4"
ARCHIVE = ROOT / ".local/scratch/astra-b2/h224-device-plan-v1/h224-device-plan-v4.tar"
PREFIX = "h224-device-plan-v4"
# Preserve the reviewed packaging dependency, excluding unrelated concurrent
# member-weight work in the shared workspace. Never rewrite that source file.
PRESERVED_V3 = {
    "tools/k1n_config_v1.py": "9a35f5b6cfd82b53ecdabdaf03ef41c5c3539c56c36a4708c162afc09aa2051b",
}


def preserved_bytes(name):
    with tarfile.open(previous.ARCHIVE, "r") as tar:
        data = tar.extractfile(previous.PREFIX + "/" + name).read()
    k.need(hashlib.sha256(data).hexdigest() == PRESERVED_V3[name],
           "preserved v3 dependency drift: " + name)
    return data


def files():
    paths = set(previous.files()) | set(k.code_pins())
    paths.update([
        "tools/l1_rco_k1_foldpack_v4.py", "tools/l1_rco_k1_localwait_v1.py",
        "tools/l1_rco_k1_foldgate_v1.py", "tools/test_l1_rco_k1_device_plan_v1.py",
        "tools/l1_rco_k1_device_plan_v1.md",
        "tools/test_l1_rco_k1_foldseq_v1.py",
        "tools/exp156_k_sweep.py", "tools/exp115_vlm_eval.py", "tools/exp115_vlm_pilot.py",
        "tools/l1_rco_k1_allseq_v1.py", "tools/l1_rco_k1_foldpack_v3.py",
        "tools/k1_fold_dataset_v1.py", "tools/k1n_config_v1.py",
        "tools/c85r7_rco_overlay_v1.py", "tools/test_l1_rco_k1_all_v1.py",
        "tools/l1_rco_k1_all_v1.md",
        (k.BASE / "FOLDS.md").relative_to(ROOT).as_posix(),
        assets.SERVING.relative_to(ROOT).as_posix(),
    ])
    for name in paths:
        path = ROOT / name
        k.need(path.resolve().is_relative_to(ROOT) and path.is_file(), "missing package input: " + name)
        k.need(not any(t in name.lower() for t in (
            "api-key", "credential", "/relabel", "/.ssh", "/f1/", "/f2/", "/f3/")), "prohibited package input")
    return sorted(paths)


def build():
    k.s.verify_pins()
    paths = files()
    PACKAGE.mkdir(parents=True, exist_ok=True)
    existing = {p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob("*") if p.is_file()}
    k.need(existing <= set(paths) | {"INVENTORY.json", "SHA256SUMS", "environment-local.json"},
           "unexpected existing package files")
    entries = []
    for name in paths:
        dst = PACKAGE / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if name in PRESERVED_V3:
            dst.write_bytes(preserved_bytes(name))
        else:
            shutil.copyfile(ROOT / name, dst)
        entries.append(dict(path=name, bytes=dst.stat().st_size, sha256=k.sha(dst)))
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    k.write(PACKAGE / "environment-local.json", dict(python=sys.version, packages=packages,
        note="Reference only. B provisions Linux dependencies; no network/install in this bundle."))
    env = PACKAGE / "environment-local.json"
    entries.append(dict(path=env.name, bytes=env.stat().st_size, sha256=k.sha(env)))
    k.write(PACKAGE / "INVENTORY.json", dict(version="4.0.0", files=entries))
    (PACKAGE / "SHA256SUMS").write_text(
        "".join(e["sha256"] + "  " + e["path"] + "\n" for e in entries), encoding="utf-8", newline="\n")
    with tarfile.open(ARCHIVE, "w") as tar:
        for name in [e["path"] for e in entries] + ["INVENTORY.json", "SHA256SUMS"]:
            tar.add(PACKAGE / name, arcname=PREFIX + "/" + name, recursive=False)
    digest = k.sha(ARCHIVE)
    ARCHIVE.with_suffix(".tar.sha256").write_text(digest + "  " + ARCHIVE.name + "\n", encoding="ascii")
    return dict(status="PASS", files=len(entries), bytes=ARCHIVE.stat().st_size, sha256=digest)


def verify(root):
    return previous.verify(root)


def verify_archive(archive=ARCHIVE):
    archive = Path(archive)
    k.need(archive.with_suffix(".tar.sha256").read_text().strip() == k.sha(archive) + "  " + archive.name,
           "archive hash mismatch")
    with tarfile.open(archive, "r") as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        k.need(len(set(names)) == len(names) and all(m.isfile() for m in members), "unsafe archive members")
        inv = json.load(tar.extractfile(PREFIX + "/INVENTORY.json"))
        expected = {PREFIX + "/" + e["path"] for e in inv["files"]} | {
            PREFIX + "/INVENTORY.json", PREFIX + "/SHA256SUMS"}
        k.need(set(names) == expected, "archive inventory mismatch")
        for entry in inv["files"]:
            name = PREFIX + "/" + entry["path"]
            with tar.extractfile(name) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            k.need(tar.getmember(name).size == entry["bytes"] and digest == entry["sha256"],
                   "archive member drift: " + name)
    return dict(status="PASS", files=len(inv["files"]), sha256=k.sha(archive))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["build", "verify", "verify-archive"])
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    result = build() if args.action == "build" else (
        verify_archive() if args.action == "verify-archive" else verify(args.root))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
