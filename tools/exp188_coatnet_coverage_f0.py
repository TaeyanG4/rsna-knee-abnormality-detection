"""v2.0.0: explicit CoAtNet preset and immutable historical access.

Changelog: retire patched producers, pilots, queue generation and timing gates;
scientific helpers remain, new execution uses the shared registered lifecycle.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import os
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "training/src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools import exp181_coatnet_f0_bridge as bridge
from tools import exp180_k12_coverage as audit
from exp188_coverage import coverage_centres, eligible_centres, independent_groups, metadata_mask_sha
from raptor_common import study_rng, train_centers

VERSION = "2.0.0"
ARM = "COVERAGE"
EXP = ROOT / "artifacts/experiments/EXP188_COATNET_COVERAGE_F0"
PROTOCOL = EXP / "PROTOCOL.md"
PROTOCOL_DIR = EXP / "protocol-v1"
METADATA = PROTOCOL_DIR / "coverage-metadata.json"
REPLAY = PROTOCOL_DIR / "cpu-replay.json"
MANIFEST = PROTOCOL_DIR / "manifest-COVERAGE.json"
VALIDATION = PROTOCOL_DIR / "validation.json"
QUEUE = PROTOCOL_DIR / "queue-exp188.json"
COLLECTION = EXP / "collection.json"
SCREEN = EXP / "screen/result-COVERAGE-vs-CONTROL.json"
SOURCE = ["tools/exp188_coatnet_coverage_f0.py", "training/src/exp188_coverage.py",
          "tools/exp180_k12_coverage.py",
          "artifacts/experiments/EXP188_COATNET_COVERAGE_F0/PROTOCOL.md",
          "artifacts/experiments/EXP188_COATNET_COVERAGE_F0/protocol-v1/coverage-metadata.json",
          "artifacts/experiments/EXP188_COATNET_COVERAGE_F0/protocol-v1/cpu-replay.json"]
WEAK = ("Lateral Meniscus", "PF OA", "Lateral OA", "Synovitis")


def digest(path):
    return bridge.sha256(path)






def spec(_arm):
    if _arm != ARM:
        raise ValueError("only the registered EXP188 COVERAGE arm is allowed")
    return dict(dataset="exp188_coverage.CoverageStudyDataset",
                augmentation=dict(name="coverage-stratified-K12", training_only=True,
                                  metadata_sha256=digest(METADATA), replay_sha256=digest(REPLAY),
                                  definition="one uniform independently available group centre, then uniform without-replacement fill"))


def run_rel(_arm):
    spec(_arm)
    return "artifacts/experiments/EXP188_COATNET_COVERAGE_F0/COVERAGE"


def dataset(_arm):
    spec(_arm)
    from exp188_coverage import CoverageStudyDataset
    return CoverageStudyDataset




def load_data():
    folds, cohort = audit.load_rosters(audit.ROSTERS)
    train, val = bridge.fold0_roster()
    if set(train) | set(val) != cohort or len(train) != bridge.N_TRAIN:
        raise ValueError("fold-0 EXP099 population differs")
    flags = audit.read_series_flags(audit.SERIES, cohort)
    _, data, provenance = audit.load_exp143(audit.E143, cohort, flags)
    return train, {uid: data[uid] for uid in train}, provenance


def slot_shares(centers, mask, slots):
    """Share of all K draws credited only to independent complete triplets."""
    counts = Counter()
    for raw in centers:
        c = int(raw)
        if c < 0:
            continue
        idx = audit.slot_index(c, slots)
        if idx is None:
            continue
        slot = slots[idx]
        if (slot["series"] is not None and not slot["fallback"]
                and slot["start"] < c < slot["stop"] - 1
                and bool(mask[c - 1:c + 2].all())):
            counts[slot["group"]] += 1
    return counts


def all_slot_shares(centers, slots):
    counts = Counter()
    for raw in centers:
        if int(raw) >= 0:
            idx = audit.slot_index(int(raw), slots)
            if idx is not None:
                counts[slots[idx]["group"]] += 1
    return counts


def replay(train, data):
    totals = {name: dict(bags=0, omitting_any=0, duplicates=0, empty=0, boundary=0,
                         opportunities=Counter(), omissions=Counter(), centres=Counter(), slots=Counter())
              for name in ("CONTROL", ARM)}
    for uid in train:
        mask, slots = data[uid]
        state = audit.availability(mask, slots)
        for epoch in range(bridge.EPOCHS):
            for name, centers in (("CONTROL", train_centers(mask, len(mask), 12, study_rng(42, epoch, uid))),
                                  (ARM, coverage_centres(mask, slots, 12, study_rng(42, epoch, uid)))):
                event, _, omitted = audit.audit_bag(centers, mask, slots, state)
                t = totals[name]
                t["bags"] += 1
                t["omitting_any"] += event["bags_omitting_any"]
                t["duplicates"] += event["duplicate_centres"]
                t["empty"] += event["empty_windows"]
                t["boundary"] += event["boundary_windows"]
                t["omissions"].update(omitted)
                t["opportunities"].update(g for g, reason in state.items() if reason == "available")
                t["centres"].update(slot_shares(centers, mask, slots))
                t["slots"].update(all_slot_shares(centers, slots))
    output = {}
    for name, t in totals.items():
        bags = t["bags"]
        if bags != len(train) * bridge.EPOCHS or t["duplicates"] or t["empty"]:
            raise ValueError("replay count or centre uniqueness failed")
        output[name] = dict(bags=bags, omitted_any=t["omitting_any"], omission_rate=t["omitting_any"] / bags,
            duplicate_centres=t["duplicates"], empty_windows=t["empty"], boundary_windows=t["boundary"],
            available_opportunities=dict(t["opportunities"]), omissions=dict(t["omissions"]),
            omission_rate_by_group={g: t["omissions"][g] / t["opportunities"][g]
                                    for g in audit.GROUPS if t["opportunities"][g]},
            independent_centre_share={g: t["centres"][g] / (bags * 12) for g in audit.GROUPS},
            slot_centre_share={g: t["slots"][g] / (bags * 12) for g in audit.GROUPS})
    if output[ARM]["omitted_any"]:
        raise ValueError("coverage replay omitted an independently available group")
    return dict(status="REPLAYED_NO_GPU", version=VERSION, seed=42, epochs=bridge.EPOCHS,
        k=12, train_studies=len(train), rng="study_rng(seed, epoch, uid) separately reset for paired arms",
        arms=output,
        independent_centre_share_delta={g: output[ARM]["independent_centre_share"][g]
                                        - output["CONTROL"]["independent_centre_share"][g]
                                        for g in audit.GROUPS},
        slot_centre_share_delta={g: output[ARM]["slot_centre_share"][g]
                                 - output["CONTROL"]["slot_centre_share"][g]
                                 for g in audit.GROUPS})





























def registration(*_args, **_kwargs):
    from training.arms import coatnet_history
    return coatnet_history.read_registration(MANIFEST)


def verified_candidate():
    from training.arms import coatnet_history
    return coatnet_history.verified_arm(MANIFEST)


def main(argv=None):
    from tools import coatnet_history_cli
    return coatnet_history_cli.main(argv, preset='coverage-f0-s42', protocol=MANIFEST, candidate=ARM)


if __name__ == '__main__':
    sys.exit(main())
