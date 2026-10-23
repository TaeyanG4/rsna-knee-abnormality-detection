"""EXP135 local external caches. Run: python -m tools.extcache.cli --help"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import SOURCES
from .common import ROOT, add_series, build_production, disk, gen_dir, read_index, require_space, sha256, verify
from .sources import (fastmri_dcm_series, fastmri_sc_series, iter_fastmri_dcm,
                      iter_fastmri_sc, iter_kneemri, iter_mrnet, oai_tasks, read_oai_task)


def _done(source, key, root):
    row = next((r for r in read_index(source, root) if r["series_key"] == key), None)
    return bool(row and (gen_dir(source, root) / row["npz_file"]).is_file()
                and sha256(gen_dir(source, root) / row["npz_file"]) == row["npz_sha256"])


def _read_oai_safe(task):
    # B 2026-10-08: one undecodable OAI series (e.g. mixed slice shapes) is logged and skipped,
    # never fatal for the rolling build; its raw tar is never a deletion candidate (not indexed).
    try:
        return task, read_oai_task(task), None
    except Exception as e:  # noqa: BLE001
        return task, None, f"{type(e).__name__}: {e}"


def _log_failure(source, task, error, root):
    path = gen_dir(source, root) / "failures.csv"
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        if new:
            f.write("series_key,archive,error\n")
        f.write(f"{task[2]},{task[0]},\"{error.replace(chr(34), chr(39))}\"\n")


def generate(source: str, root: Path = ROOT, visit: str = "V00", limit: int = 0, workers: int = 1):
    require_space(root)
    built = skipped = failed = 0
    if source == "oai":
        tasks = oai_tasks(root, visit, limit)
        tasks = [t for t in tasks if not _done(source, t[2], root)]
        skipped = len(oai_tasks(root, visit, limit)) - len(tasks)
        # Batches bound decoded volumes and pending IPC results to worker count.
        failures = gen_dir(source, root) / "failures.csv"
        failed_keys = ({line.split(",", 1)[0] for line in failures.read_text(encoding="utf-8").splitlines()[1:]}
                       if failures.exists() else set())
        tasks = [t for t in tasks if t[2] not in failed_keys]
        results = (ProcessPoolExecutor(max_workers=workers) if workers > 1 else None)
        try:
            for start in range(0, len(tasks), workers):
                chunk = tasks[start:start + workers]
                for task, series, error in (results.map(_read_oai_safe, chunk) if results else map(_read_oai_safe, chunk)):
                    if error:
                        _log_failure(source, task, error, root); failed += 1
                    else:
                        add_series(source, series, root); built += 1
        finally:
            if results:
                results.shutdown()
    elif source == "fastmri_sc":
        for name, archive, data in iter_fastmri_sc(root):
            key = f"FMRI_SC_{Path(name).stem}"
            if _done(source, key, root):
                skipped += 1; continue
            add_series(source, fastmri_sc_series(name, data, archive), root); built += 1
            if limit and built >= limit:
                break
    elif source == "fastmri_dcm":
        for key, archive, group in iter_fastmri_dcm(root):
            parts = Path(key).parts
            series_key = f"FMRI_DCM_{parts[-2]}_{parts[-1]}"
            if _done(source, series_key, root):
                skipped += 1; continue
            add_series(source, fastmri_dcm_series(key, archive, group), root); built += 1
            if limit and built >= limit:
                break
    elif source in ("kneemri", "mrnet"):
        iterator = iter_kneemri(root) if source == "kneemri" else iter_mrnet(root)
        for series in iterator:
            if _done(source, series.meta["series_key"], root):
                skipped += 1; continue
            add_series(source, series, root); built += 1
            if limit and built >= limit:
                break
    else:
        raise ValueError(source)
    return dict(source=source, built=built, skipped=skipped, failed=failed, indexed=len(read_index(source, root)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for command in ("gen", "prod", "verify"):
        p = sub.add_parser(command)
        p.add_argument("--source", required=True, choices=SOURCES)
        p.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
        if command == "gen":
            p.add_argument("--visit", default="V00")
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--workers", type=int, default=1)
        elif command == "prod":
            p.add_argument("--shard-size", type=int, default=16)
    p = sub.add_parser("disk")
    p.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.command == "disk":
        result = disk(a.root)
    elif a.command == "gen":
        if a.workers < 1 or a.workers > 4 or a.limit < 0:
            ap.error("workers must be 1..4 and limit nonnegative")
        result = generate(a.source, a.root, a.visit, a.limit, a.workers)
    elif a.command == "prod":
        if not 1 <= a.shard_size <= 32:
            ap.error("shard-size must be 1..32")
        result = build_production(a.source, a.root, a.shard_size)
    else:
        result = verify(a.source, a.root)
    print(json.dumps(result, indent=2))
    if a.command == "verify" and result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
