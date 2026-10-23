"""Checked same-volume directory renames with an immutable rollback journal."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPING = {
    "data": "storage/competition", "test": "storage/test",
    "cache": "storage/cache/derived", "dicom_cache": "storage/cache/dicom",
    "pixel_cache": "storage/cache/pixels", "processed": "storage/processed",
    "labels": "storage/labels", "gold": "storage/gold", "external": "storage/external",
    "weights": "storage/weights", "experiments": "artifacts/experiments",
    "oof": "artifacts/oof", "predictions": "artifacts/predictions", "logs": "artifacts/logs",
    "kaggle_notebooks": "research/notebooks", "reports": "research/reports",
    "metadata": "research/metadata", "snapshots": "research/snapshots",
    "kaggle_assets": "kaggle/assets", "kaggle_submission": "kaggle/packages",
    "submissions": "kaggle/submissions", "src": "training/src", "scripts": "training/runners",
}
JOURNAL = ROOT / ".local/migrations/layout-v2/moves.json"


def checked(relative: str) -> Path:
    p = (ROOT / relative).resolve()
    if p == ROOT or not p.is_relative_to(ROOT) or p.drive != ROOT.drive:
        raise ValueError(f"Unsafe move target {p}")
    return p


def fingerprint(directory: Path) -> list[dict]:
    # Directory rename preserves all descendants. Verify direct members and small
    # metadata; don't read hundreds of GB of DICOM/cache just to rename a directory.
    result = []
    for p in sorted(directory.iterdir()):
        row = {"name": p.name, "directory": p.is_dir()}
        if p.is_file():
            row["size"] = p.stat().st_size
            if p.stat().st_size <= 10_000_000:
                row["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        result.append(row)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    args = ap.parse_args()
    if args.rollback:
        rows = json.loads(JOURNAL.read_text(encoding="utf-8"))
        for row in reversed(rows):
            src, dest = checked(row["source"]), checked(row["destination"])
            if dest.exists() and not src.exists():
                os.rename(dest, src)
            elif dest.exists() and src.exists():
                raise RuntimeError("Rollback conflict; preserve both directories for inspection")
        old = ROOT / ".local/migrations/layout-v2/workspace-before-move.toml"
        (ROOT / "config/workspace.toml").write_bytes(old.read_bytes())
        print("Directory layout rolled back; source edits are retained in the backup archive.")
        return
    if JOURNAL.exists():
        raise RuntimeError("Move journal already exists; inspect it rather than rerunning migration")
    rows = []
    for source, destination in MAPPING.items():
        src, dest = checked(source), checked(destination)
        if not src.is_dir() or dest.exists():
            raise RuntimeError(f"Migration precondition failed: {src} -> {dest}")
        rows.append({"source": source, "destination": destination, "fingerprint": fingerprint(src), "moved": False})
    if not args.apply:
        print(json.dumps({"mode": "dry-run", "moves": [{k:r[k] for k in ['source','destination']} for r in rows]}, indent=2))
        return
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    config = ROOT / "config/workspace.toml"
    (JOURNAL.parent / "workspace-before-move.toml").write_bytes(config.read_bytes())
    JOURNAL.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row in rows:
        src, dest = checked(row["source"]), checked(row["destination"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dest)
        if fingerprint(dest) != row["fingerprint"]:
            raise RuntimeError(f"Post-move fingerprint mismatch: {dest}")
        row["moved"] = True
        JOURNAL.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    text = config.read_text(encoding="utf-8").replace("layout_version = 1", "layout_version = 2")
    for alias, target in MAPPING.items():
        text = text.replace(f'{alias} = "{alias}"', f'{alias} = "{target}"')
    config.write_text(text, encoding="utf-8")
    print(json.dumps({"moved": len(rows), "verified": len(rows), "journal": str(JOURNAL)}))


if __name__ == "__main__":
    main()
