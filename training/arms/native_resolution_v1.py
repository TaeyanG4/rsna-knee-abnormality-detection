"""Native-detail MIL hook v1.0.0 (2026-10-13), H-B-147.

Changelog: paired 224/320 from raw-derived 448 at unchanged 140/100 mm;
two-study microbatches, effective batch eight, channels-last on both sides.
Never accepts the legacy 336 raster as a source for a 320 experiment.
"""
import copy
from training.arms import mil_screen_hooks_v1 as h

VERSION = "1.0.0"
WORK = ".local/scratch/astra-b3/H-B-147"
CORPUS = WORK + "/native448"
CONTRACT = "hb147_raptor64_448_c140_s02_v1"
SOURCES = ("training/arms/native_resolution_v1.py", "training/arms/mil_screen_hooks_v2.py",
           "training/runners/mil_screen_hooks_v2.py", "tools/hb147_build_v1.py",
           "tools/hb147_native_cache_v1.py", "tools/hb147_native_audit_v1.py",
           "training/src/raptor_corpus.py", "tools/hb147_go_v1.py", WORK + "/PROTOCOL.md")


def scale_windows(vol, centers, resolution):
    h.need(vol.shape == (64, 448, 448) and vol.dtype == h.np.uint8,
           "native448 required; never upscale the old 336 cache")
    h.need(resolution in (224, 320) and len(centers) in (12, 24, 36), "unregistered windows")
    out = h.np.empty((len(centers), 3, resolution, resolution), h.np.float32)
    for j, center in enumerate(centers):
        if center == -1:
            out[j] = 0
            continue
        center = max(1, min(int(center), 62))
        image = vol[center] if j % 2 == 0 else vol[center, 64:384, 64:384]
        plane = h.torch.from_numpy(image[None].astype(h.np.float32) / 255.)
        out[j] = h.torch.nn.functional.interpolate(plane[None], size=(resolution, resolution),
            mode="bilinear", align_corners=False)[0].numpy()
    return out


class NativeWindows(h.MILWindows):
    def __init__(self, inner, resolution):
        self.__dict__.update(inner.__dict__)
        self.resolution = resolution

    def __getitem__(self, index):
        if self.corpus is None:
            self.corpus = h.common.Corpus(self.corpus_dir)
        h.need(self.corpus.contract == CONTRACT and self.corpus.spec["img"] == 448,
               "native source contract differs")
        uid, epoch = self.ids[index], int(self.epoch.item())
        rng = h.common.study_rng(self.seed, epoch, uid) if self.train else None
        centers = h.common.select_centers(self.corpus.mask(uid), 64, self.k, self.corpus.spec,
                                         "within-slot-valid", rng)
        x = scale_windows(self.corpus.volume(uid), centers, self.resolution)
        if self.train:
            intensity = h.common.study_rng(self.seed, epoch, uid, "intensity").random()
            x = h.np.clip(x * (1 + (intensity - .5) * .20), 0, 1)
            x = h.augment_windows(x, centers, rng, self.corpus.spec)
        mean = h.np.array([.485, .456, .406], h.np.float32).reshape(3, 1, 1)
        std = h.np.array([.229, .224, .225], h.np.float32).reshape(3, 1, 1)
        x = (x - mean) / std
        h.need(x.dtype == h.np.float32 and h.np.isfinite(x).all(), "invalid native windows")
        return h.torch.from_numpy(x), h.torch.from_numpy(self.labels[uid]), uid


class NativeArm(h.Arm):
    name = "HB147_NATIVE"
    sources = SOURCES
    screen = "HB147_NATIVE_RESOLUTION_V1"
    microbatch = 2

    def validate_recipe(self, spec):
        d = spec.document
        h.need(d["seed"] in (42, 43) and d["fold"] == 0 and spec.run_kind == "member", "pilot identity differs")
        old = h.control_spec(d["seed"]).document
        c = copy.deepcopy(d["config"])
        res = c["windows"]["res"]
        h.need(res in (224, 320), "unregistered resolution")
        h.need(d["arm"] == f'HB147_NATIVE{res}_S{d["seed"]}_F0', "arm name differs")
        h.need(c["windows"].pop("native_source") == CONTRACT, "native contract missing")
        h.need(c["sampler"].pop("microbatch") == 2 and c["updates"]["accum"] == 4,
               "effective batch must remain eight")
        h.need(c["model"].pop("memory_format") == "channels_last", "memory format differs")
        h.need(c["roots"]["corpus"] == CORPUS and c["roots"]["manifest"] == CORPUS + "/manifest.json",
               "source cache differs")
        c["roots"]["corpus"] = old["config"]["roots"]["corpus"]
        c["roots"]["manifest"] = old["config"]["roots"]["manifest"]
        c["windows"]["res"], c["updates"]["accum"] = 224, 1
        h.need(c == old["config"], "scientific recipe differs beyond registered common changes and resolution")
        for key in ("family", "seed", "fold", "target_environment", "outputs", "attempt"):
            h.need(d[key] == old[key], "shared identity differs: " + key)
        expected = dict(old["inputs"]["data"])
        del expected[old["config"]["roots"]["manifest"]]
        expected[CORPUS + "/manifest.json"] = d["inputs"]["data"][CORPUS + "/manifest.json"]
        h.need(d["inputs"]["data"] == expected and d["inputs"]["rosters"] == old["inputs"]["rosters"]
               and d["inputs"]["controls"] == {}, "data identity differs")
        h.need(set(d["inputs"]["sources"]) == h.expected_sources(self, h.control_spec(d["seed"])),
               "source inventory differs")
        h.need(all(d["inputs"]["sources"][p] == digest for p, digest in old["inputs"]["sources"].items()),
               "shared source bindings differ")
        h.need(d["inputs"]["sources"][h.control_path(d["seed"])] == h.control_spec(d["seed"]).sha256,
               "base control binding differs")
        return d["config"]

    def dataset(self, dataset, spec, stage, *, train):
        return NativeWindows(dataset, spec.document["config"]["windows"]["res"])

    def attach(self, model, spec, stage):
        return model.to(memory_format=h.torch.channels_last)

    def forward(self, model, x, extra):
        h.need(extra is None, "unexpected metadata")
        # Inference also chunks at two studies; no cross-study operation or BN.
        if not h.torch.is_grad_enabled() and len(x) > self.microbatch:
            return h.torch.cat([self.forward(model, p, None) for p in x.split(self.microbatch)])
        flat = x.flatten(0, 1).contiguous(memory_format=h.torch.channels_last)
        return model(flat.reshape(x.shape))

    def provenance(self, spec):
        return dict(native_source=CONTRACT, resolution=spec.document["config"]["windows"]["res"],
                    microbatch=2, accumulation=4, memory_format="channels_last",
                    controls="fresh native448 224; legacy HB122 not causally paired")


ARM = NativeArm()


def validate_data(spec, stage, corpus_dir, *, engineering=False):
    """Engineering path validates metadata/hashes, never opens label rows."""
    c = ARM.validate_recipe(spec)
    manifest = h.read(h.Path(corpus_dir) / "manifest.json")
    h.need(manifest["finished"] and manifest["contract"] == CONTRACT and
           manifest["shape_total"] == [4410, 64, 448, 448], "native cache incomplete")
    h.need(manifest["source_manifest_sha256"] == "f3afc9ab81fe068fe3ef3ac31f3f7a64d5bffb821594d8cb30e60329624b838e",
           "native source ancestry differs")
    for key, path in (("builder_sha256", "tools/hb147_native_cache_v1.py"),
                      ("raw_builder_sha256", "training/src/raptor_corpus.py")):
        h.need(manifest[key] == spec.document["inputs"]["sources"][path], "cache builder binding differs")
    if not engineering:
        return h.validate_data_with(ARM, spec, stage, corpus_dir)
    h.life.validate_inputs(spec, stage)
    for path, digest in spec.document["inputs"]["sources"].items():
        h.need(h.sha(h.ROOT / path) == digest, "executing source differs")
    h.need(h.sha(h.Path(corpus_dir) / "manifest.json") == spec.inventory[c["roots"]["manifest"]],
           "mounted cache differs")
    for name in ("ids", "masks"):
        h.need(h.sha(h.Path(corpus_dir) / (name + ".npy")) == manifest[name + "_sha256"], "cache index differs")
    corpus = h.common.Corpus(corpus_dir)
    r = c["roots"]
    canonical = h.read(h.Path(stage) / r["canonical"])
    with (h.Path(stage) / r["groups"]).open(newline="", encoding="utf-8") as stream:
        groups = list(h.csv.DictReader(stream))
    train, val = h.validate_membership(corpus.ids.tolist(), h.read(h.Path(stage) / r["ordered"]), groups,
                                     canonical["train"] + canonical["val"], r["order_policy"])
    h.need((len(train),len(val)) == (r["n_train"],r["n_val"]) and h.uid_hash(train) == r["train_sha256"]
           and h.uid_hash(val) == r["val_sha256"], "roster differs")
    labels = {u: h.np.full(12, .5, h.np.float32) for u in train + val}
    return corpus, train, val, labels, h.np.asarray(c["labels"]["pos_weight"], h.np.float32)
