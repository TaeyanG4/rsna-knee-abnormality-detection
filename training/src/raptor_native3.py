"""Native-adjacent triplet inputs (H-B-25): each window = [native c-1, c, native c+1] instead of [cache c-1, c, cache c+1].

The 64-slot cache (raptor64_336_c140_s02) samples each series with np.linspace over 2-98 % of its slices, so two neighbouring cache
planes are usually 1.5-3 native slices apart (D-A-011). This module keeps everything of that contract - series selection, IOP/IPP
order, span rounding, the 64 picks, the per-slot p2/p98 window, the 140 mm centred crop, 336 px INTER_AREA, uint8 truncation, the
64-entry mask, slot layout and the within-slot-valid centre choice - and only changes where the two outer channels of a window come
from: the slices directly before and after the picked centre slice in the series' sorted (physical) order.

Cache layout (contract raptor64n3_336_c140_s02, built by training/src/raptor_native3_corpus.py): uint8 (N, 64, 3, 336, 336);
plane 1 of slot position j is byte-identical to the 64-slot cache plane j; planes 0 / 2 are the native previous / next slice of
that pick. Rules:
  * normalisation: the slot's own p2/p98 window, computed exactly as before over the 64 picked slices only (repeats included);
    the neighbours are mapped with that same window (so the centre channel and the base cache stay byte-identical and the three
    channels share one intensity scale). Neighbour pixels outside the window are clipped like every other slice.
  * crop/resize: the neighbour's own PixelSpacing (median of the series when <= 0), 140 mm centred crop, INTER_AREA to 336 px.
  * edges: a pick at native index 0 (or n-1) has no previous (next) slice -> that channel replicates the centre plane; same when
    a neighbour fails to decode (counted in the build receipts). A centre that failed to decode stays all-zero in all 3 planes.
  * mask: positive pixel sum of the centre plane (= the 64-slot cache mask); window validity therefore stays the cache rule.

Nothing here is executed at import (no file access, no torch import). raptor_common.CONTRACTS is NOT extended: DicomCorpus would
otherwise render this contract as a plain 64-plane volume. The serving/verification renderer prepare_study_native3 mirrors
raptor_dicom.prepare_study and adds the neighbours. The training dataset (StudyWindows with windows_from_native3) lives in
raptor_native3_data.py as a module-level class so spawn-started DataLoader workers can unpickle it.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

CONTRACT = "raptor64n3_336_c140_s02"
BASE_CONTRACT = "raptor64_336_c140_s02"
CORPUS_DIRNAME = "RAPTOR64N3_336_C140_S02"
PLANES = 3
PLANE_ORDER = ["native_prev", "centre", "native_next"]
NEIGHBOUR_RULE = ("planes 0/2 = the slice directly before/after the picked slice in the series' IOP/IPP sort order "
                  "(the base contract's order); replicate the centre at series edges or on neighbour decode failure")
NORMALISATION_RULE = ("slot p2/p98 computed over the 64-slot picks only (repeats included, as raptor64); neighbours mapped "
                      "with the same window, clip, uint8 truncation")
ADAPTER_VERSION = "raptor-native3-v1"


def spec_for(base_spec):
    """Cache spec of the native3 contract: the base spec plus the plane description (depth/slots stay the 64-slot ones)."""
    spec = dict(base_spec)
    spec.update(planes=PLANES, plane_order=list(PLANE_ORDER), neighbours=NEIGHBOUR_RULE, neighbour_normalisation=NORMALISATION_RULE,
                base_contract=BASE_CONTRACT)
    return spec


def neighbour_indices(index, n):
    """(previous, next) native indices of a pick at sorted index `index` in a series of n slices; the centre at the edges."""
    index, n = int(index), int(n)
    if n <= 0 or not 0 <= index < n:
        raise ValueError(f"pick index {index} outside a series of {n} slices")
    return (index - 1 if index > 0 else index), (index + 1 if index + 1 < n else index)


def is_native3_spec(spec):
    return int(spec.get("planes", 1)) == PLANES and list(spec.get("plane_order", [])) == PLANE_ORDER


# ------------------------------------------------------------------ model inputs
def windows_from_native3(vol3, centers, res):
    """(K,3,res,res) float32 in [0,1] from a (depth,3,H,W) native3 volume; same clamp/scale/interpolation as raptor_common.windows_from."""
    import torch
    import torch.nn.functional as F
    if vol3.ndim != 4 or vol3.shape[1] != PLANES:
        raise ValueError(f"expected a (depth, 3, H, W) native3 volume, got {tuple(vol3.shape)}")
    depth = vol3.shape[0]
    out = np.empty((len(centers), 3, res, res), np.float32)
    for j, c in enumerate(centers):
        if c == -1:
            out[j] = 0
            continue
        c = max(1, min(int(c), depth - 2))
        tri = np.asarray(vol3[c]).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res or t.shape[-2] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear", align_corners=False)[0]
        out[j] = t.numpy()
    return out


def prepare_input_native3(corpus, uid, k_eval, res, norm, window_policy="within-slot-valid"):
    """Serving counterpart of raptor_runtime.prepare_input for a native3 corpus (cache or DicomCorpusNative3)."""
    from raptor_common import select_centers
    from raptor_runtime import normalize_input
    mask = corpus.mask(uid)
    centers = select_centers(mask, corpus.depth, k_eval, corpus.spec, policy=window_policy)
    return normalize_input(windows_from_native3(corpus.volume(uid), centers, res), norm)


# ------------------------------------------------------------------ DICOM render (serving / independent verification)
def _render(pixels, spacing, loq, hiq, crop_mm, img):
    """raptor_dicom.prepare_study per-slice math (normalise with the slot window, centred physical crop, INTER_AREA, uint8)."""
    import cv2
    normalized = np.clip((pixels - loq) / (hiq - loq + 1e-6), 0, 1)
    height, width = normalized.shape
    crop = min(int(round(float(crop_mm) / max(spacing, 1e-3))), min(height, width))
    if crop <= 0:
        raise ValueError("empty physical crop")
    y0, x0 = (height - crop) // 2, (width - crop) // 2
    resized = cv2.resize(normalized[y0:y0 + crop, x0:x0 + crop], (img, img), interpolation=cv2.INTER_AREA)
    return (resized * 255).astype(np.uint8)


def prepare_study_native3(uid, rows, source, spec, split="test"):
    """raptor_dicom.prepare_study (same series pick, order, picks, window, failure policy for picks) plus the native neighbours.

    Returns volume (depth, 3, img, img) uint8, mask (depth,) uint8 and a receipt. A neighbour that fails to decode is replaced by
    the centre and counted (receipt neighbour_decode_failures); a pick that fails raises like prepare_study.
    """
    from raptor_dicom import ADAPTER_VERSION as BASE_ADAPTER, SERIES, DicomPreparationError, _decode, _flag, _ordered_members, pick_series
    depth, img = int(spec["depth"]), int(spec["img"])
    volume = np.zeros((depth, PLANES, img, img), np.uint8)
    receipt = {"uid": str(uid), "split": str(split), "adapter": ADAPTER_VERSION, "base_adapter": BASE_ADAPTER, "status": "preparing",
               "slots": [], "acquired_slices": 0, "metadata_rows": len(rows), "neighbour_decode_failures": 0, "edge_replications": 0,
               "mask_semantics": "positive normalized pixel sum of the centre plane"}
    offset, used = 0, set()
    try:
        for slot_index, (plane, fluid, count) in enumerate(spec["slots"]):
            sr = {"slot": slot_index, "plane": plane, "fluid": fluid, "width": count, "offset": offset, "status": "metadata_absent",
                  "instance_order_fallbacks": 0}
            receipt["slots"].append(sr)
            record = pick_series(rows, plane, fluid, used)
            if record is None:
                offset += count
                continue
            series = str(record[SERIES]); used.add(series)
            sr.update(series=series, status="acquired", protocol_flag=_flag(record.get("Fluid_Sensitive")))
            members = source.members(split, str(uid), series)
            if not members:
                sr["status"] = "acquired_files_missing"
                raise RuntimeError(f"metadata names acquired series without DICOM files: {uid}/{series}")
            files, median_spacing = _ordered_members(source, members, sr)
            n = len(files)
            low = int(n * spec["span"][0]); high = max(int(n * spec["span"][1]) - 1, low)
            selected = np.linspace(low, high, count).round().astype(int) if n > 1 else np.zeros(count, int)
            arrays = {}
            for index in sorted(set(map(int, selected))):
                member = files[min(index, n - 1)][0]
                try:
                    arrays[index], _ = _decode(source, member)
                except Exception as exc:
                    sr.update(status="decode_failed", failed_source=member.path)
                    raise RuntimeError(f"DICOM pixels failed: {member.path}: {exc}") from exc
            loq, hiq = np.percentile(np.concatenate([arrays[int(i)].ravel() for i in selected]), [2.0, 98.0])
            neighbours = {}
            for index in selected:
                for nb in neighbour_indices(min(int(index), n - 1), n):
                    if nb not in arrays and nb not in neighbours:
                        try:
                            neighbours[nb], _ = _decode(source, files[nb][0])
                        except Exception:
                            neighbours[nb] = None
            sr.update(files=n, slices=selected.tolist(), normalization=[float(loq), float(hiq)], median_spacing=median_spacing)

            def plane_of(nb):
                pixels = arrays.get(nb, neighbours.get(nb))
                if pixels is None:
                    return None
                spacing = files[nb][1]
                return _render(pixels, spacing if spacing > 0 else median_spacing, loq, hiq, spec["crop_mm"], img)

            for index in selected:
                q = min(int(index), n - 1)
                centre = plane_of(q)
                volume[offset, 1] = centre
                for channel, nb in zip((0, 2), neighbour_indices(q, n)):
                    if nb == q:
                        volume[offset, channel] = centre; receipt["edge_replications"] += 1
                        continue
                    rendered = plane_of(nb)
                    if rendered is None:
                        volume[offset, channel] = centre; receipt["neighbour_decode_failures"] += 1
                    else:
                        volume[offset, channel] = rendered
                offset += 1
            sr.update(status="rendered", rendered=count)
            receipt["acquired_slices"] += count
        if offset != depth:
            raise ValueError(f"slot depth {offset} does not match {depth}")
        mask = (volume[:, 1].reshape(depth, -1).sum(1) > 0).astype(np.uint8)
        receipt.update(status="metadata_absent" if not used else "rendered", acquired_slots=len(used), positive_mask_slices=int(mask.sum()),
                       volume_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(volume)).cast("B")).hexdigest(),
                       mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
        volume.setflags(write=False); mask.setflags(write=False)
        return volume, mask, receipt
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise DicomPreparationError(f"study {uid}: {exc}", receipt) from exc


def dicom_corpus_class():
    """DicomCorpusNative3: raptor_dicom.DicomCorpus with prepare_study_native3 and the native3 spec (lazy class, no import side effects)."""
    from raptor_dicom import DicomCorpus, DicomPreparationError, _contract_spec

    class DicomCorpusNative3(DicomCorpus):
        def _initialize(self, ids, rows, source, contract, split):
            if contract != CONTRACT:
                raise ValueError(f"DicomCorpusNative3 serves only {CONTRACT}")
            super()._initialize(ids, rows, source, BASE_CONTRACT, split)
            self.contract = CONTRACT
            self.spec = spec_for(_contract_spec(BASE_CONTRACT))
            self.manifest.update(contract=CONTRACT, spec=self.spec, adapter=ADAPTER_VERSION)

        def _ensure(self, uid):
            uid = str(uid)
            if uid not in self.rows:
                raise KeyError(uid)
            if self._cached_uid != uid:
                self._cached_uid = None; self._cached = None
                try:
                    volume, mask, receipt = prepare_study_native3(uid, self.rows[uid], self.source, self.spec, self.split)
                except DicomPreparationError as exc:
                    self.receipts[uid] = exc.receipt
                    raise
                self.receipts[uid] = receipt
                self._cached_uid = uid; self._cached = (volume, mask)
            return self._cached

    return DicomCorpusNative3
