"""Corpus contracts added without editing raptor_common.py (H-B-22, D-B-019).

raptor_common.py is bound by sha256 in the running EXP094/EXP099 protocols, so a new contract cannot be added there
while those queues run. Importing this module adds the entries below to raptor_common.CONTRACTS in place (the same dict
object every consumer reads: raptor_corpus.main, raptor_dicom._contract_spec / DicomCorpus). Nothing else changes:
series selection, slice order, span rounding, percentile window, crop, INTER_AREA resize and uint8 truncation are the
raptor64 code paths, only the slot table differs.

raptor80_336_c140_s02 = the public dense corpus layout (dreaddevelopment/raptor-knee-finespacing, v9 weights):
80 slices = 22 sagittal fluid-sensitive / 18 sagittal other / 15 coronal fluid-sensitive / 10 coronal other / 15 axial,
span 2-98 % of each series, 140 mm physical crop, 336 px. Downstream readers (Corpus, select_centers, StudyWindows,
raptor_runtime.predict) take the slot table from the cache manifest spec, so they need no change.

Importing twice is a no-op; a conflicting existing entry raises instead of being overwritten.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import raptor_common  # noqa: E402

SLOTS80 = [("Sagittal", 1, 22), ("Sagittal", 0, 18), ("Coronal", 1, 15), ("Coronal", 0, 10), ("Axial", -1, 15)]

EXT_CONTRACTS = {
    # public "Knee MRI - Fine Spacing Corpus" layout (raptor_ft_coatnet_v9_full.pt; public 0.932 at 78 windows per its description)
    "raptor80_336_c140_s02": dict(slots=SLOTS80, img=336, crop_mm=140.0, span=(0.02, 0.98)),
}
# Reference only (not used by any loader): the public dense arm reads every position of the 80-slice volume.
EXT_PUBLIC_ARMS = {
    "finespacing-v9": dict(file="dreaddevelopment__raptor-knee-finespacing/raptor_ft_coatnet_v9_full.pt",
                           contract="raptor80_336_c140_s02", k_eval=78),
}


def register(contracts=None):
    """Add EXT_CONTRACTS to raptor_common.CONTRACTS (or the given mapping); return the names added or already present."""
    target = raptor_common.CONTRACTS if contracts is None else contracts
    names = []
    for name, spec in EXT_CONTRACTS.items():
        if name in target and target[name] != spec:
            raise RuntimeError(f"contract {name!r} already registered with a different spec")
        target.setdefault(name, dict(spec))
        names.append(name)
    return names


def depth(name):
    return sum(int(s[2]) for s in raptor_common.CONTRACTS[name]["slots"])


register()
