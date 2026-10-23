"""Per-study DICOM header facts for validation diagnostics (scanner group, sex, laterality).

Reads one header per series straight from the competition ZIP (no pixels) and writes
storage/labels/derived_meta/study_meta.csv. Scanner grouping follows the public metadata
probes (Manufacturer x field strength, ~13 groups) plus a finer fingerprint
(model + software + coil). Nothing here feeds training unless a run opts in.

  python training/src/study_meta.py
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import re
import sqlite3
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raptor_common import ROOT, UID  # noqa: E402

OUT = ROOT / "storage" / "labels" / "derived_meta" / "study_meta.csv"
FIELDS = ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "SoftwareVersions", "ReceiveCoilName",
          "PatientSex", "Laterality", "ImagingFrequency"]


class IndexedZip(zipfile.ZipFile):
    def _RealGetContents(self):
        self.start_dir = os.fstat(self.fp.fileno()).st_size


def zip_info(path, info_json):
    info = zipfile.ZipInfo(path)
    for key, value in json.loads(info_json).items():
        setattr(info, key, value)
    return info


def normalize_manufacturer(value):
    v = (value or "").lower()
    for key, name in (("siemens", "Siemens"), ("philips", "Philips"), ("ge ", "GE"), ("ge_", "GE"), ("general electric", "GE"),
                      ("canon", "Canon"), ("toshiba", "Canon"), ("hitachi", "Hitachi"), ("fujifilm", "Fujifilm"), ("united imaging", "UIH"), ("uih", "UIH")):
        if key in v or v.startswith(key.strip()):
            return name
    return value.strip()[:20] if value else "unknown"


def normalize_field(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "unknown"
    return "3T" if f >= 2.5 else "1.5T" if f >= 1.2 else "low"


def normalize_laterality(value):
    v = (value or "").strip().upper()
    return "R" if v in ("R", "RIGHT") else "L" if v in ("L", "LEFT") else "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import pydicom
    db = sqlite3.connect(ROOT / ".local" / "kneeops" / "app.sqlite")
    arc = json.loads(db.execute("SELECT value FROM meta WHERE key='data_library'").fetchone()[0])["archives"][0]
    zip_path = Path(arc["path"].replace("workspace://", str(ROOT) + "/"))
    con = sqlite3.connect(f"file:{arc['index']}?mode=ro", uri=True)
    # First file of every series, with the ZipInfo that belongs to that same file.
    rows = con.execute("SELECT split, uid, series, path, info FROM files WHERE split<>'' ORDER BY split, uid, series, path").fetchall()
    by_study, seen = collections.OrderedDict(), set()
    for split, uid, series, path, info in rows:
        if (uid, series) in seen:
            continue
        seen.add((uid, series))
        by_study.setdefault((split, uid), []).append((path, info))
    items = list(by_study.items())
    if a.limit:
        items = items[:a.limit]
    out_rows = []
    with IndexedZip(zip_path) as z:
        for n, ((split, uid), members) in enumerate(items):
            values = collections.defaultdict(collections.Counter)
            for path, info in members:
                try:
                    with z.open(zip_info(path, info)) as f:
                        d = pydicom.dcmread(io.BytesIO(f.read(96 * 1024)), stop_before_pixels=True, force=True)
                except Exception:
                    continue
                for field in FIELDS:
                    v = getattr(d, field, None)
                    if v not in (None, ""):
                        values[field][str(v)] += 1
            pick = {f: (values[f].most_common(1)[0][0] if values[f] else "") for f in FIELDS}
            manu, field = normalize_manufacturer(pick["Manufacturer"]), normalize_field(pick["MagneticFieldStrength"])
            fingerprint = "|".join([manu, pick["ManufacturerModelName"].strip(), re.sub(r"\s+", " ", pick["SoftwareVersions"])[:40], pick["ReceiveCoilName"].strip()])
            out_rows.append({UID: uid, "split": split, "series": len(members), "manufacturer": manu, "model": pick["ManufacturerModelName"].strip(),
                             "field": field, "scanner_group": f"{manu}_{field}", "fingerprint": fingerprint, "software": pick["SoftwareVersions"][:60],
                             "coil": pick["ReceiveCoilName"], "sex": pick["PatientSex"] or "unknown", "laterality": normalize_laterality(pick["Laterality"])})
            if (n + 1) % 500 == 0:
                print(f"{n + 1}/{len(items)}", flush=True)
    import pandas as pd
    df = pd.DataFrame(out_rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} studies")
    print(df["scanner_group"].value_counts().to_string())
    print("sex", df["sex"].value_counts().to_dict(), "| laterality", df["laterality"].value_counts().to_dict(), "| fingerprints", df["fingerprint"].nunique())


if __name__ == "__main__":
    main()
