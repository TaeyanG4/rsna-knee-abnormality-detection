"""extcache_rolling v1.0.0 — rolling external-data pipeline: GENERAL -> PRODUCTION -> verify -> delete verified raw archives.

User approval (Taeyang, 2026-10-08 17:2x KST: "<owner remark removed>"): a raw archive may be deleted once its GENERAL
cache is verified (every npz re-hashed, every PRODUCTION study re-derived byte-identically) and the source is
re-downloadable. Only paths that tools/extcache verify lists as safe_to_delete_candidates are removed, only under
storage/external_data or storage/downloads, and every deletion is written to a receipt first.

  python tools/extcache_rolling.py --source oai --visit V00 --workers 6           # build + verify + delete
  python tools/extcache_rolling.py --source oai --visit V00 --no-delete           # build + verify only
  python tools/extcache_rolling.py --delete-redundant PATH --because "REASON"     # e.g. a zip whose extracted copy exists
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "artifacts/experiments/EXP135_EXTCACHE/deletions"
ALLOWED = (ROOT / "storage/external_data", ROOT / "storage/downloads")
FLOOR = 100 * 1024 ** 3


def _cli(*args):
    cmd = [sys.executable, "-m", "tools.extcache.cli", *args]
    print("+", " ".join(cmd), flush=True)
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode:
        print(out.stdout[-3000:], out.stderr[-3000:], sep="\n", flush=True)
        raise SystemExit(f"extcache {args[0]} failed ({out.returncode})")
    return out.stdout


def _json_tail(text):
    i = text.rfind("\n{")
    return json.loads(text[i + 1:] if i >= 0 else text[text.find("{"):])


def _allowed(p: Path) -> bool:
    p = p.resolve()
    return any(str(p).lower().startswith(str(a.resolve()).lower() + os.sep) for a in ALLOWED)


def _receipt(kind, items, extra):
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    path = RECEIPTS / f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    body = dict(schema="extcache-deletion-v1", kind=kind, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                approval="Taeyang 2026-10-08 17:2x KST rolling deletion approved", items=items, **extra)
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return path


def delete(paths, kind, extra):
    items = []
    for p in map(Path, paths):
        if not _allowed(p):
            raise SystemExit(f"refusing to delete outside storage/external_data|downloads: {p}")
        if p.exists():
            items.append(dict(path=str(p), bytes=p.stat().st_size if p.is_file() else None, dir=p.is_dir()))
    receipt = _receipt(kind, items, extra)  # receipt before deleting
    freed = 0
    for it in items:
        p = Path(it["path"])
        if it["dir"]:
            shutil.rmtree(p)
        else:
            freed += it["bytes"] or 0
            p.unlink()
    print(json.dumps(dict(receipt=str(receipt), deleted=len(items), freed_bytes=freed), indent=2), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source")
    ap.add_argument("--visit")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-delete", action="store_true")
    ap.add_argument("--delete-redundant", nargs="+")
    ap.add_argument("--because")
    a = ap.parse_args(argv)
    if a.delete_redundant:
        if not a.because:
            raise SystemExit("--because is required")
        delete(a.delete_redundant, "redundant", dict(reason=a.because))
        return
    if shutil.disk_usage(ROOT).free < FLOOR:
        raise SystemExit("disk floor: less than 100 GiB free")
    gen = ["gen", "--source", a.source, "--workers", str(a.workers)] + (["--visit", a.visit] if a.visit else [])
    _cli(*gen)
    _cli("prod", "--source", a.source)
    report = _json_tail(_cli("verify", "--source", a.source))
    summary = {k: report[k] for k in ("general_series", "production_studies", "general_bytes", "production_bytes")}
    print(json.dumps(dict(verify=summary, errors=report["errors"][:20],
                          candidates=len(report["safe_to_delete_candidates"])), indent=2), flush=True)
    if report["errors"]:
        raise SystemExit("verify reported errors; nothing deleted")
    if not a.no_delete and report["safe_to_delete_candidates"]:
        delete(report["safe_to_delete_candidates"], f"verified_{a.source}", dict(verify=summary))


if __name__ == "__main__":
    main()
