"""EXP155 Amendment B1 OAI labels. CPU only; aggregate output only.

The EXP136 clinical receipt authenticates the three source tables. This module
never prints or writes a participant-level QA table.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re

from tools import extlabels
from tools.exp155_oai_data import check_cache
from tools.exp143_v2 import ROOT, TARGETS, sha256

VERSION = "1.1.0"
SCHEMA = "exp155-oai-labels-v2"
TARGETS_OAI = ("PF OA", "Lateral OA", "Synovitis")
DEFAULT_CACHE = ROOT / "storage/cache/derived/EXP155_OAI_RAPTOR96_384_C140_S02"
DEFAULT_RECEIPT = ROOT / "storage/labels/external/OAI_receipt.json"
DEFAULT_OUTPUT = ROOT / "storage/labels/derived_constellation/exp155_oai_masked_v2.csv"
SOURCE_NAMES = ("kMRI_SQ_MOAKS_BICL00.txt", "kMRI_POMA_IncOA_MOAKS_BICL.txt",
                "kMRI_POMA_TKR_MOAKS_BICL.txt")


def write_new_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def clinical_inputs(receipt_path):
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("contract") != "EXP136-v1" or receipt.get("tool_sha256") != sha256(ROOT / "tools/extlabels.py"):
        raise ValueError("EXP136 clinical receipt/tool changed")
    bindings = receipt.get("inputs_sha256", {})
    selected = {}
    for name in SOURCE_NAMES:
        matches = [(Path(path), digest) for path, digest in bindings.items()
                   if Path(path).name.casefold() == name.casefold()]
        if len(matches) != 1:
            raise ValueError("missing/ambiguous receipt input: " + name)
        path, digest = matches[0]
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or sha256(path) != digest:
            raise ValueError("clinical source hash changed: " + name)
        selected[name] = (path, digest)
    return selected


def oa_v2(scores, target):
    """Only compartment cartilage decides; subgrade 2/3 is a proxy."""
    if scores is None:
        return None
    parts = [extlabels.cartilage_parts(scores.get("MCM" + region))
             for region in extlabels.CART_REGIONS[target]]
    if any(part is not None and part[1] >= 2 for part in parts):
        return 1
    if all(part == (0, 0) for part in parts):
        return 0
    return None


def synovitis_v2(primary, poma):
    """Observed primary MSYIC wins; POMA fills only a missing target."""
    if primary is not None and primary.get("MSYIC") is not None:
        value = extlabels.threshold(primary["MSYIC"], range(4), 2)
        return value, "moaks" if value is not None else "unknown"
    value = extlabels.threshold(poma.get("MSYIC"), range(4), 2) if poma is not None else None
    return value, "poma" if value is not None else "unknown"


def select_sources(inputs):
    moaks = extlabels.read_oai(inputs[SOURCE_NAMES[0]][0], "moaks")
    poma = [record for name in SOURCE_NAMES[1:]
            for record in extlabels.read_oai(inputs[name][0], "poma")]
    primary, _ = extlabels.select_readings(moaks, extlabels.MOAKS_ORDER)
    fallback, _ = extlabels.select_readings(poma, ("INCOA", "TKR"))
    return primary, fallback


def aggregate_qa(labels, cache_n, rsna_n=4347):
    if rsna_n != 4347 or cache_n < len(labels):
        raise ValueError("registered RSNA count/cache size changed")
    counts = {}
    shares = {}
    for target in TARGETS_OAI:
        positive = sum(row[target] == 1 for row in labels.values())
        negative = sum(row[target] == 0 for row in labels.values())
        observed = positive + negative
        counts[target] = dict(positive=positive, negative=negative,
                              unknown=cache_n - observed, observed=observed)
        shares[target] = dict(rsna=rsna_n / (rsna_n + observed),
                              oai=observed / (rsna_n + observed),
                              rsna_observed_cells=rsna_n, oai_observed_cells=observed)
    n = len(labels)
    return dict(schema="exp155-oai-label-qa-v2", version=VERSION, cache_n=cache_n,
                admitted_oai_n=n, excluded_all_unknown_n=cache_n - n,
                rsna_n=rsna_n, mixed_n=rsna_n + n,
                updates_per_epoch=math.ceil((rsna_n + n) / 16),
                total_updates=16 * math.ceil((rsna_n + n) / 16),
                counts=counts, source_share_before_pos_weight=shares,
                coefficient_definition="observed source cells/(4347+OAI observed cells), fixed denominator 12")


def build_labels(args):
    cache = Path(args.oai_cache)
    out = Path(args.out)
    provenance_path = out.with_suffix(".provenance.json")
    qa_path = out.with_suffix(".qa.json")
    if any(p.exists() for p in (out, provenance_path, qa_path)):
        raise FileExistsError("v2 label output exists; choose a fresh version")
    manifest = check_cache(cache, full_hash=False)
    inputs = clinical_inputs(args.receipt)
    primary, poma = select_sources(inputs)
    rendered = []
    mapped = {}
    for uid in manifest["ids"]:
        if not re.fullmatch(r"OAI_[0-9]+_[RL]_V00", uid):
            raise ValueError("cache contains nonbaseline OAI identity")
        m = primary[uid].scores if uid in primary else None
        f = poma[uid].scores if uid in poma else None
        values = {target: oa_v2(m, target) for target in TARGETS_OAI[:2]}
        values["Synovitis"], _ = synovitis_v2(m, f)
        if not any(value is not None for value in values.values()):
            continue
        mapped[uid] = values
        row = {"study_key": uid, **{target: "" for target in TARGETS},
               **{"mask_" + target: 0 for target in TARGETS}}
        for target, value in values.items():
            if value is not None:
                row[target] = value
                row["mask_" + target] = 1
        rendered.append(row)
    if not rendered:
        raise ValueError("no observed OAI cells")
    qa = aggregate_qa(mapped, len(manifest["ids"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["study_key", *TARGETS, *("mask_" + target for target in TARGETS)]
    with out.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rendered)
    write_new_json(qa_path, qa)
    provenance = dict(schema=SCHEMA, version=VERSION, targets=list(TARGETS_OAI),
        other_targets_masked=True, mapping="Amendment B1 MOAKS cartilage v2 + target-level POMA Synovitis",
        reader_precedence="EXP136 selected MOAKS; POMA IncOA then TKR, R1 then R2, version/filename/content",
        cache_manifest_sha256=sha256(cache / "manifest.json"),
        exp136_receipt_sha256=sha256(args.receipt),
        clinical_inputs_sha256={name: digest for name, (_, digest) in inputs.items()},
        csv_sha256=sha256(out), qa_sha256=sha256(qa_path),
        rows=len(rendered))
    write_new_json(provenance_path, provenance)
    return dict(status="COMPLETE", labels=str(out), sha256=provenance["csv_sha256"],
                qa=str(qa_path), **qa)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oai-cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return ap


if __name__ == "__main__":
    print(json.dumps(build_labels(parser().parse_args()), indent=2))
