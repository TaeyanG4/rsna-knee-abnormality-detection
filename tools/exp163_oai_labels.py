"""EXP163 extended OAI labels. CPU only; aggregate output only.

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
from tools.exp155_oai_labels_v2 import clinical_inputs, select_sources, oa_v2, synovitis_v2
from tools.exp143_v2 import ROOT, TARGETS, sha256

VERSION = "1.1.0"
SCHEMA = "exp163-oai-labels-v1.1"
TARGETS_OAI = ("ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
               "Lateral OA", "PF OA", "Effusion", "Synovitis")
ADDED_MOAKS = ("Medial Meniscus", "Lateral Meniscus", "Effusion")
UNSUPERVISED = ("MCL", "Baker's", "Contusion", "Fracture")
DEFAULT_CACHE = ROOT / "storage/cache/derived/EXP155_OAI_RAPTOR96_384_C140_S02"
DEFAULT_RECEIPT = ROOT / "storage/labels/external/OAI_receipt.json"
DEFAULT_OUTPUT = ROOT / "storage/labels/derived_constellation/exp163_oai_masked_v1_1.csv"
SOURCE_NAMES = ("kMRI_SQ_MOAKS_BICL00.txt", "kMRI_POMA_IncOA_MOAKS_BICL.txt",
                "kMRI_POMA_TKR_MOAKS_BICL.txt")


def write_new_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def map_extended(primary, poma):
    """Preserve B1's three heads; use EXP163 thresholds for added MOAKS heads.

    Added non-OA heads use a selected POMA record only if the whole primary
    MOAKS reading is absent. A partial/invalid selected reading cannot be
    silently reinterpreted as a negative using a different assessment.
    Medial OA uses the same cartilage-only quality rule as the existing OA
    heads, with record-level POMA fallback when primary is absent.
    """
    primary_map = extlabels.map_moaks(primary) if primary is not None else None
    fallback_map = extlabels.map_moaks(poma, poma=True) if poma is not None else None
    values = {target: None for target in TARGETS_OAI}
    for target in ("Lateral OA", "PF OA"):
        values[target] = oa_v2(primary, target)
    values["Medial OA"] = oa_v2(primary if primary is not None else poma, "Medial OA")
    values["Synovitis"], _ = synovitis_v2(primary, poma)
    for target in ADDED_MOAKS:
        values[target] = (primary_map if primary_map is not None else fallback_map or {}).get(target)
    acl_source = primary if primary is not None else poma
    acl_grade = extlabels.grade(acl_source.get("MACLTR"), range(3)) if acl_source is not None else None
    values["ACL"] = None if acl_grade is None or acl_grade == 1 else int(acl_grade == 2)
    return values


def aggregate_qa(labels, cache_n, rsna_n=4347):
    if rsna_n != 4347 or cache_n < len(labels):
        raise ValueError("registered RSNA count/cache size changed")
    counts = {}
    shares = {}
    for target in TARGETS:
        positive = sum(row.get(target) == 1 for row in labels.values())
        negative = sum(row.get(target) == 0 for row in labels.values())
        observed = positive + negative
        counts[target] = dict(positive=positive, negative=negative,
                              unknown=cache_n - observed,
                              admitted_unknown=len(labels) - observed,
                              observed=observed)
        shares[target] = dict(rsna=rsna_n / (rsna_n + observed),
                              oai=observed / (rsna_n + observed),
                              rsna_observed_cells=rsna_n, oai_observed_cells=observed,
                              exceeds_30_percent=observed / (rsna_n + observed) > .30)
    n = len(labels)
    exposure = {target: dict(oai_positive_cells_per_epoch=counts[target]["positive"],
                             oai_negative_cells_per_epoch=counts[target]["negative"],
                             oai_observed_cells_per_epoch=counts[target]["observed"],
                             rsna_observed_cells_per_epoch=rsna_n,
                             oai_positive_fraction_of_mixed_observed_cells=counts[target]["positive"] / (rsna_n + counts[target]["observed"]),
                             oai_negative_fraction_of_mixed_observed_cells=counts[target]["negative"] / (rsna_n + counts[target]["observed"]))
                for target in TARGETS}
    return dict(schema="exp163-oai-label-qa-v1.1", version=VERSION, cache_n=cache_n,
                admitted_oai_n=n, excluded_all_unknown_n=cache_n - n,
                rsna_n=rsna_n, mixed_n=rsna_n + n,
                updates_per_epoch=math.ceil((rsna_n + n) / 16),
                total_updates=16 * math.ceil((rsna_n + n) / 16),
                counts=counts, per_epoch_exposure=exposure, source_share_before_pos_weight=shares,
                coefficient_definition="observed source cells/(4347+OAI observed cells), before RSNA-only positive weighting, fixed denominator 12",
                oai_targets=list(TARGETS_OAI), masked_targets=list(UNSUPERVISED),
                flagged_targets=[t for t in TARGETS_OAI if shares[t]["exceeds_30_percent"]])


def build_labels(args):
    cache = Path(args.oai_cache)
    out = Path(args.out)
    provenance_path = out.with_suffix(".provenance.json")
    qa_path = out.with_suffix(".qa.json")
    if any(p.exists() for p in (out, provenance_path, qa_path)):
        raise FileExistsError("EXP163 label output exists; choose a fresh version")
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
        values = map_extended(m, f)
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
        other_targets_masked=True, mapping="EXP163 v1.1: ACL MACLTR 2 positive, 0 negative, 1/invalid unknown; OAI Baker's always masked; B1 cartilage-only PF/lateral, synovitis target fallback; EXP136 meniscus/effusion, medial cartilage-only",
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
