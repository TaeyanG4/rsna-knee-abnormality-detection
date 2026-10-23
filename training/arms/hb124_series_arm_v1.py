"""H-B-124 shared-control hook adapter v1.0.0 (2026-10-13).

One recipe extension; existing context/control sources remain unchanged.
Acquisition metadata only. Panel Sag0 copies retain their real source series.
"""
import json
from pathlib import Path

import torch

from training.arms import mil_screen_hooks_v1 as hooks
from training.src.hb124_series_context_v1 import POLICY, SeriesProvenance, WithinSeriesContext

VERSION = "1.0.0"
TRAIN = "storage/cache/derived/RAPTOR64_336_C140_S02"
PANEL = "storage/cache/derived/EXT_OAI_RAPTOR64_336_C140_v2"
CORPORA = {
    TRAIN: dict(manifest="f3afc9ab81fe068fe3ef3ac31f3f7a64d5bffb821594d8cb30e60329624b838e",
                picks="62a1c63b90e5baa8404cbf501e69b915a13e2910af034c11c93b7b452aaa7cb1"),
    PANEL: dict(manifest="c503168d7d4ba2edd45a6a3ce0b138afa8dcfc079e0b5d12a891a25489e43fa9",
                picks="7a6c466205157c5d4c26cc32820dbd2fcc2edfc0d2d61dad23eaa20c3f55f8f5"),
}


class BoundProvenance(SeriesProvenance):
    def __init__(self, corpus, picks_path, *, panel=False):
        super().__init__(corpus, picks_path)
        self.copies = 0
        if panel:
            # The cache's documented Sag0 fallback is a resampling of slot 0,
            # not a second acquisition. Keep source indices/duplicate handling.
            sizes = [int(s[2]) for s in corpus.spec["slots"]]
            with Path(picks_path).open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    source, copy = row["slots"][:2]
                    if copy is None or copy.get("fallback") != "copy_of_sagittal_fluid1":
                        continue
                    hooks.need(source is not None and source["files"] > 0
                               and copy["series"] == source["series"] + "#sag0_fallback_copy"
                               and copy["group_key"] == source["group_key"]
                               and copy["files"] == source["files"]
                               and copy["source_sequence"] == source["source_sequence"],
                               "panel copied-series provenance differs")
                    series, _ = self.by_uid[row["uid"]]
                    series[sizes[0]:sum(sizes[:2])] = series[0]
                    self.copies += 1


class SeriesWindows(hooks.CenterWindows):
    """Spawn-safe metadata wrapper; same shared epoch tensor and image stream."""
    def __init__(self, inner, picks_path, picks_sha256, *, panel=False):
        super().__init__(inner)
        self.picks_path, self.picks_sha256 = str(picks_path), picks_sha256
        self.panel, self.provenance = panel, None

    def __getitem__(self, index):
        x, y, uid, centers = super().__getitem__(index)
        if self.provenance is None:
            self.provenance = BoundProvenance(self.inner.corpus, self.picks_path, panel=self.panel)
            hooks.need(self.provenance.sha256 == self.picks_sha256, "series sidecar changed")
        return x, y, uid, self.provenance.windows(uid, centers.numpy())


def collate_series(rows):
    x, y, ids = hooks.collate([r[:3] for r in rows])
    metadata = {key: torch.stack([r[3][key] for r in rows]) for key in ("series", "position", "valid")}
    return x, y, ids, metadata


class SeriesContextArm(hooks.Arm):
    name, section, key = "HB124_SERIES", "model", "series_context_v1"
    screen = "HB124_SERIES_CONTEXT_V1"
    sources = ("training/arms/hb124_series_arm_v1.py", "training/src/hb124_series_context_v1.py",
               "tools/hb124_hook_build_v1.py", "tools/hb124_hook_smoke_v1.py",
               *tuple(path + "/picks.jsonl" for path in CORPORA), PANEL + "/manifest.json")
    collate_fn = staticmethod(collate_series)

    def contract(self, seed):
        return dict(version=VERSION, policy=POLICY, hidden=32, initialization="zero-output-residual",
                    placement="encode-context-finding-MIL", centers="shared-CenterWindows-exact-replay",
                    panel_copies="checked-Sag0-source-series", corpora=CORPORA)

    def dataset(self, dataset, spec, stage, *, train):
        self.validate_recipe(spec)
        manifest = hooks.sha(Path(dataset.corpus_dir) / "manifest.json")
        matches = [p for p, pins in CORPORA.items() if pins["manifest"] == manifest]
        hooks.need(len(matches) == 1, "unregistered series corpus")
        path = matches[0]
        hooks.need(not train or path == TRAIN, "training corpus differs")
        picks = Path(stage) / path / "picks.jsonl"
        hooks.need(spec.inventory[path + "/picks.jsonl"] == CORPORA[path]["picks"]
                   and hooks.sha(picks) == CORPORA[path]["picks"], "bound series sidecar differs")
        return SeriesWindows(dataset, picks, CORPORA[path]["picks"], panel=path == PANEL)

    def attach(self, model, spec, stage):
        self.validate_recipe(spec)
        hooks.need(not hasattr(model, "series_context"), "series context already attached")
        model.series_context = WithinSeriesContext(model.backbone.num_features, hidden=32)
        return model

    def probe(self, device):
        return dict(series=torch.full((1, 1), -1, dtype=torch.int64, device=device),
                    position=torch.zeros(1, 1, device=device),
                    valid=torch.zeros(1, 1, dtype=torch.bool, device=device))

    def move(self, extra, device):
        hooks.need(len(extra) == 1 and isinstance(extra[0], dict)
                   and set(extra[0]) == {"series", "position", "valid"}, "series metadata required")
        return {key: value.to(device, non_blocking=True) for key, value in extra[0].items()}

    def forward(self, model, x, extra):
        hooks.need(extra is not None and all(tuple(v.shape) == tuple(x.shape[:2]) for v in extra.values()),
                   "series metadata shape differs")
        return model.head(model.series_context.encoded(model.encode(x), extra))

    def provenance(self, spec):
        return dict(series_context=self.contract(spec.document["seed"]), series_metadata="acquisition-only")


ARM = SeriesContextArm()
