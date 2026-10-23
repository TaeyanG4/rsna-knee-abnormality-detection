"""Plan OAI image archive selection; this module never downloads anything.

Example: ``python tools/oai_plan.py --visit V00 --mode moaks_knees
--metadata image03.txt --package-file package_file_metadata_1249722.txt
--moaks kMRI_SQ_MOAKS_BICL00.txt --inflight-bytes 0``.

The mandatory ``--inflight-bytes`` includes all other downloads already running
or queued on the destination disk. Set it to zero only after checking that none
are in flight. A complete archive in ``--download-root`` is deducted from the
incremental requirement; partial files are not deducted.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Iterable


DEFAULT_SERIES = ("SAG_IW_TSE", "COR_IW_TSE", "AX_MPR", "COR_MPR")
MIN_FREE_BYTES = 100 * 1024**3
CSV_FIELDS = ("study_key", "ID", "side", "series", "image_file", "FILE_SIZE",
              "incremental_bytes", "download_alias")


def _rows(path: Path, delimiter: str) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"empty or duplicate table header: {path}")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"malformed row width: {path}")
            yield {str(k).strip().strip('"'): v.strip() for k, v in row.items()}


def _need_fields(row: dict[str, str], fields: Iterable[str], source: Path) -> None:
    missing = sorted(set(fields) - row.keys())
    if missing:
        raise ValueError(f"{source} missing columns: {', '.join(missing)}")


def _side(value: str) -> str:
    v = value.strip().upper()
    if v in ("1", "1: RIGHT", "RIGHT", "R"):
        return "R"
    if v in ("2", "2: LEFT", "LEFT", "L"):
        return "L"
    raise ValueError(f"unknown knee side: {value!r}")


def _clinical_number(value: str) -> float | None:
    """Read a numeric SAS code, including '2: Description', but not '.A: Missing'."""
    head = value.split(":", 1)[0].strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", head):
        return None
    number = float(head)
    return number if math.isfinite(number) and number >= 0 else None


def clinical_knees(path: Path, visit: str, mode: str) -> set[tuple[str, str]]:
    """Find assessable knees, not positive knees, in the matching visit table."""
    if mode == "all_knees":
        return set()
    out: set[tuple[str, str]] = set()
    inspected = False
    for row in _rows(path, "|"):
        if not inspected:
            _need_fields(row, ("ID", "SIDE", "VERSION"), path)
            if mode == "kl_knees":
                _need_fields(row, (f"{visit}XRKL",), path)
            inspected = True
        knee_id = row["ID"].strip()
        if not knee_id.isdecimal():
            continue
        if mode == "kl_knees":
            assessable = _clinical_number(row[f"{visit}XRKL"]) in (0, 1, 2, 3, 4)
        else:
            # MOAKS valid grades end at 8. Scores above 9 are coded missing
            # values, not evidence that a knee was assessed.
            assessable = any((number := _clinical_number(v)) is not None and number <= 9
                             for k, v in row.items() if k.startswith(f"{visit}M") and
                             k not in (f"{visit}MCMNTS", f"{visit}MTCMNTS"))
        if assessable:
            out.add((knee_id, _side(row["SIDE"])))
    if not inspected:
        raise ValueError(f"no clinical records: {path}")
    if not out:
        raise ValueError(f"no assessable {mode} knees in {path}")
    return out


def package_sizes(path: Path) -> dict[str, tuple[int, str]]:
    out: dict[str, tuple[int, str]] = {}
    inspected = False
    for row in _rows(path, ","):
        if not inspected:
            _need_fields(row, ("NDA_S3_URL", "FILE_SIZE", "DOWNLOAD_ALIAS"), path)
            inspected = True
        url = row["NDA_S3_URL"]
        if not url.startswith("s3://"):
            raise ValueError(f"invalid package URL in {path}")
        try:
            size = int(row["FILE_SIZE"])
        except ValueError as exc:
            raise ValueError(f"invalid file size in {path}") from exc
        if size <= 0 or not row["DOWNLOAD_ALIAS"]:
            raise ValueError(f"missing positive size or alias in {path}")
        value = (size, row["DOWNLOAD_ALIAS"])
        if url in out and out[url] != value:
            raise ValueError(f"conflicting package entries in {path}")
        out[url] = value
    if not out:
        raise ValueError(f"empty package file: {path}")
    return out


def _existing_complete(root: Path | None, alias: str, size: int) -> bool:
    relative = Path(alias.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or ":" in alias or not relative.parts:
        raise ValueError("unsafe download alias in package metadata")
    if root is None:
        return False
    base = root.resolve()
    target = (base / relative).resolve()
    if not target.is_relative_to(base):
        raise ValueError("download alias escapes download root")
    return target.is_file() and target.stat().st_size == size


def select_images(metadata: Path, packages: dict[str, tuple[int, str]],
                  visit: str, series: tuple[str, ...], knees: set[tuple[str, str]] | None,
                  download_root: Path | None = None) -> list[dict[str, str | int]]:
    out: list[dict[str, str | int]] = []
    seen: dict[tuple[str, str, str], str] = {}
    inspected = False
    for row in _rows(metadata, "\t"):
        if not inspected:
            _need_fields(row, ("collection_id", "src_subject_id", "visit",
                               "image_description", "image_file"), metadata)
            inspected = True
        if row["collection_id"] == "collection_id" or row["visit"].upper() != visit:
            continue
        description = row["image_description"].strip().upper()
        match = re.fullmatch(r"(.+)_(RIGHT|LEFT)", description)
        if not match or match.group(1) not in series:
            continue
        knee_id = row["src_subject_id"].strip()
        if not knee_id.isdecimal():
            raise ValueError("selected image has a nonnumeric OAI subject ID")
        side = _side(match.group(2))
        if knees is not None and (knee_id, side) not in knees:
            continue
        image_file = row["image_file"].strip()
        if image_file not in packages:
            raise ValueError("selected image missing package size")
        key = (knee_id, side, match.group(1))
        if key in seen:
            if seen[key] != image_file:
                raise ValueError("multiple archives for the same knee and series")
            continue
        seen[key] = image_file
        size, alias = packages[image_file]
        increment = 0 if _existing_complete(download_root, alias, size) else size
        out.append({"study_key": f"OAI_{knee_id}_{side}_{visit}", "ID": knee_id,
                    "side": "RIGHT" if side == "R" else "LEFT", "series": key[2],
                    "image_file": image_file, "FILE_SIZE": size,
                    "incremental_bytes": increment, "download_alias": alias})
    if not inspected or not out:
        raise ValueError(f"no selected {visit} images in {metadata}")
    order = {name: i for i, name in enumerate(series)}
    out.sort(key=lambda r: (str(r["study_key"]), order[str(r["series"])],
                            str(r["image_file"])))
    return out


def check_budget(rows: list[dict[str, str | int]], disk_root: Path,
                 inflight_bytes: int) -> dict[str, int]:
    if inflight_bytes < 0:
        raise ValueError("inflight bytes must be nonnegative")
    free = shutil.disk_usage(disk_root).free
    incremental = sum(int(row["incremental_bytes"]) for row in rows)
    projected = free - inflight_bytes - incremental
    if projected < MIN_FREE_BYTES:
        raise ValueError(f"disk budget rejected: {projected:,} projected free bytes "
                         f"< {MIN_FREE_BYTES:,} required on {disk_root}")
    return {"series_count": len(rows),
            "knee_count": len({str(row["study_key"]) for row in rows}),
            "total_bytes": sum(int(row["FILE_SIZE"]) for row in rows),
            "incremental_bytes": incremental, "inflight_bytes": inflight_bytes,
            "currently_free_bytes": free, "projected_free_bytes": projected}


def write_plan(rows: list[dict[str, str | int]], out_dir: Path, stem: str) -> tuple[Path, Path]:
    csv_path, txt_path = out_dir / f"{stem}.csv", out_dir / f"{stem}.txt"
    if csv_path.exists() or txt_path.exists():
        raise FileExistsError(f"refusing to overwrite {csv_path} or {txt_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    created_csv = created_txt = False
    try:
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            created_csv = True
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with txt_path.open("x", encoding="utf-8", newline="") as stream:
            created_txt = True
            for row in rows:
                stream.write(str(row["image_file"]) + "\n")
    except Exception:
        if created_csv:
            csv_path.unlink()
        if created_txt:
            txt_path.unlink()
        raise
    return csv_path, txt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visit", required=True, type=str.upper)
    parser.add_argument("--mode", required=True,
                        choices=("moaks_knees", "kl_knees", "all_knees"))
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--package-file", required=True, type=Path)
    parser.add_argument("--moaks", type=Path)
    parser.add_argument("--kl", type=Path)
    parser.add_argument("--series", nargs="+", default=list(DEFAULT_SERIES))
    parser.add_argument("--series-tag", help="short filename tag for selected series")
    parser.add_argument("--download-root", type=Path)
    parser.add_argument("--disk-root", type=Path, default=Path("H:/"))
    parser.add_argument("--inflight-bytes", type=int, required=True)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("storage/external_data/oai/selection"))
    args = parser.parse_args(argv)
    if not re.fullmatch(r"V\d{2}", args.visit):
        parser.error("visit must be V followed by two digits")
    selected_series = tuple(name.upper() for name in args.series)
    if len(set(selected_series)) != len(selected_series) or not all(
            re.fullmatch(r"[A-Z0-9_]+", item) for item in selected_series):
        parser.error("series must be distinct uppercase identifiers")
    source = args.moaks if args.mode == "moaks_knees" else args.kl
    if args.mode != "all_knees" and source is None:
        parser.error(f"--{'moaks' if args.mode == 'moaks_knees' else 'kl'} required")
    tag = args.series_tag or ("4series" if selected_series == DEFAULT_SERIES else
                              "-".join(selected_series))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        parser.error("series tag must be a safe filename component")
    knees = None if args.mode == "all_knees" else clinical_knees(source, args.visit, args.mode)
    packages = package_sizes(args.package_file)
    rows = select_images(args.metadata, packages, args.visit, selected_series,
                         knees, args.download_root)
    totals = check_budget(rows, args.disk_root, args.inflight_bytes)
    csv_path, txt_path = write_plan(rows, args.out_dir,
                                    f"{args.visit}_{args.mode}_{tag}")
    print(json.dumps({**totals, "csv": str(csv_path), "txt": str(txt_path)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
