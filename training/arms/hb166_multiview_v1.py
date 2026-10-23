"""H-B-166 multi-view finding-MIL arm v1.0.0 (2026-10-15).

One MIL bag per study = the UNCHANGED H-B-155 control full-field native448
windows (same class, same RNG streams, byte-identical pixels) PLUS compartment
crop windows from crop-cache-v220 (H-B-155 crop-arm windowing, side-v2 flag
authentication, but INDEPENDENT RNG streams so the full-field part never
changes). Every window carries a view id (0 = full field, 1 = compartment
crop); a learned view-type embedding E[2, F] (zero-initialised, 1,536
parameters for ConvNeXt-S) is added to every spatial token of that window
before the shared finding LayerNorm. The existing twelve per-target attention
heads pool over the tokens of BOTH views, so each target chooses its views.

Window budget (registered): K windows PER VIEW. Train K=12 -> 12 full + 12 crop
= 24 windows/study (backbone compute 2.0x the control's 12); validation K24 ->
24+24, K36 -> 36+36, K36flip -> 36+36 with the registered horizontal flip
applied per view using that view's own cache mask (= control / crop-arm TTA).
No other recipe change.
"""
import torch
from torch import nn

from training.arms import hb155_crops_v1 as crops
from training.arms import mil_screen_hooks_v3 as h
from training.arms import native_resolution_v2 as native

VERSION = "1.0.0"
FULL, CROP = 0, 1
VIEW_NAMES = ("full_field_native448", "compartment_crop_v220")
CROP_STREAMS = ("hb166-crop-windows", "hb166-crop-intensity")
FULL_CONTRACT = native.CONTRACT
CROP_CONTRACT = crops.SIDE_CONTRACT


class CropWindows(crops.Windows):
    """H-B-155 crop-arm windows; only the two RNG stream names differ.

    Evaluation (train=False) has no RNG, so its windows equal crops.Windows on
    the same crop corpus exactly (tested).
    """

    def __getitem__(self, index):
        if self.corpus is None:
            self.corpus = h.common.Corpus(self.corpus_dir)
        h.need(self.corpus.contract == CROP_CONTRACT and self.corpus.spec["img"] == 448,
               "H-B-166 crop view requires the side-v2 crop cache")
        uid, epoch = self.ids[index], int(self.epoch.item())
        self.last_side_unknown = crops.side_flag(self.corpus, uid)
        rng = h.common.study_rng(self.seed, epoch, uid, CROP_STREAMS[0]) if self.train else None
        centers = h.common.select_centers(self.corpus.mask(uid), 64, self.k, self.corpus.spec,
                                         "within-slot-valid", rng)
        x = native.scale_windows(self.corpus.volume(uid), centers, self.resolution)
        if self.train:
            intensity = h.common.study_rng(self.seed, epoch, uid, CROP_STREAMS[1]).random()
            x = h.np.clip(x * (1 + (intensity - .5) * .20), 0, 1)
            x = h.augment_windows(x, centers, rng, self.corpus.spec)
        mean = h.np.array([.485, .456, .406], h.np.float32).reshape(3, 1, 1)
        std = h.np.array([.229, .224, .225], h.np.float32).reshape(3, 1, 1)
        x = (x - mean) / std
        h.need(x.dtype == h.np.float32 and h.np.isfinite(x).all(), "invalid H-B-166 crop windows")
        return torch.from_numpy(x), torch.from_numpy(self.labels[uid]), uid


class MultiViewWindows(torch.utils.data.Dataset):
    """(x[2K,3,R,R], y[12], uid, views[2K]); first K = full field, last K = crop."""

    def __init__(self, full, crop):
        h.need(type(full) is crops.Windows and type(crop) is CropWindows, "registered view datasets required")
        h.need(full.ids == crop.ids and full.k == crop.k and full.train == crop.train
               and full.seed == crop.seed and full.resolution == crop.resolution, "view datasets differ")
        self.full, self.crop = full, crop

    @property
    def k(self):
        return self.full.k

    @property
    def ids(self):
        return self.full.ids

    @property
    def train(self):
        return self.full.train

    def set_epoch(self, epoch):
        self.full.set_epoch(epoch)
        self.crop.set_epoch(epoch)

    def __len__(self):
        return len(self.full)

    def __getitem__(self, index):
        xf, yf, uf = self.full[index]
        h.need(self.full.corpus.contract == FULL_CONTRACT, "full-field view must use native448")
        xc, yc, uc = self.crop[index]
        h.need(uf == uc and torch.equal(yf, yc), "view rows differ")
        views = torch.tensor([FULL] * len(xf) + [CROP] * len(xc), dtype=torch.int64)
        return torch.cat([xf, xc]), yf, uf, views


def make_dataset(full_dir, crop_dir, ids, labels, k, train, seed, resolution):
    full = crops.Windows(h.MILWindows(full_dir, ids, labels, k, train, seed), resolution)
    crop = CropWindows(h.MILWindows(crop_dir, ids, labels, k, train, seed), resolution)
    return MultiViewWindows(full, crop)


def collate(rows):
    xs, ys, ids, views = zip(*rows)
    return torch.stack(xs), torch.stack(ys), list(ids), torch.stack(views)


class MultiViewFindingMIL(h.FindingMIL):
    """FindingMIL + view_embed. forward REQUIRES per-window view ids."""

    def forward(self, x, views):
        h.need(views.shape == x.shape[:2] and views.dtype == torch.int64, "view ids must be int64[B,K]")
        tokens, mask = self.encode(x)
        per_window = tokens.shape[1] // x.shape[1]
        h.need(per_window * x.shape[1] == tokens.shape[1], "token layout differs")
        tokens = tokens + self.view_embed[views].repeat_interleave(per_window, dim=1).to(tokens.dtype)
        return self.head((tokens, mask))


def attach_views(model):
    """Convert a freshly built control FindingMIL in place; no RNG draw, zero init.

    State keys = control keys + 'view_embed'. The name does not start with
    'backbone.' so optimizer_stack puts it in the head group (lr 1e-3, wd .02).
    """
    h.need(type(model) is h.FindingMIL and not hasattr(model, "view_embed"), "control FindingMIL required")
    reference = model.clsW
    model.__class__ = MultiViewFindingMIL
    model.view_embed = nn.Parameter(torch.zeros(len(VIEW_NAMES), model.backbone.num_features,
                                                dtype=reference.dtype, device=reference.device))
    return model


class MultiViewArm:
    name = "HB166_MULTIVIEW"
    microbatch = 2
    collate_fn = staticmethod(collate)

    def forward(self, model, x, views):
        h.need(isinstance(model, MultiViewFindingMIL), "multi-view model required")
        # Inference chunks at two studies (no cross-study op, no BN), like NativeArm.
        if not torch.is_grad_enabled() and len(x) > self.microbatch:
            return torch.cat([self.forward(model, a, b) for a, b in
                              zip(x.split(self.microbatch), views.split(self.microbatch))])
        flat = x.flatten(0, 1).contiguous(memory_format=torch.channels_last)
        return model(flat.reshape(x.shape), views)


ARM = MultiViewArm()


def flip_views(x, views, ids, full_corpus, crop_corpus, k):
    """Registered K36 flip per view, each with its own cache mask/slot layout."""
    h.need(x.shape[1] == 2 * k and bool((views[:, :k] == FULL).all()) and bool((views[:, k:] == CROP).all()),
           "multi-view layout differs")
    return torch.cat([h.flip_inputs(x[:, :k], ids, full_corpus, k),
                      h.flip_inputs(x[:, k:], ids, crop_corpus, k)], dim=1)


def train_epoch(arm, model, loader, opt, sched, scaler, weights, device, amp, max_skips, counters, progress):
    """mil_screen_hooks_v3.train_epoch_accum with per-window view ids; same arithmetic."""
    progress.phase_start("training_step")
    model.train()
    opt.zero_grad(set_to_none=True)
    total, batches, durations = 0., 0, []
    lossf = nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
    for x, y, ids, views in loader:
        if device == "cuda":
            torch.cuda.synchronize()
        then = h.time.perf_counter()
        value = 0.
        for start in range(0, len(ids), arm.microbatch):
            stop = min(start + arm.microbatch, len(ids))
            with h.autocast(device, amp):
                logits = arm.forward(model, x[start:stop].to(device, non_blocking=True),
                                     views[start:stop].to(device, non_blocking=True))
                h.need(logits.shape == (stop - start, 12) and bool(torch.isfinite(logits).all()), "invalid logits")
                loss = lossf(logits.float(), y[start:stop].to(device)) * ((stop - start) / len(ids))
            h.need(bool(torch.isfinite(loss)), "nonfinite loss")
            scaler.scale(loss).backward()
            value += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=amp != "fp16")
        before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if amp == "fp16" and device == "cuda" and scaler.get_scale() < before:
            counters["skips"] += 1
            h.need(counters["skips"] <= max_skips, "scientific optimizer skip limit exceeded")
        else:
            sched.step()
        counters["updates"] += 1
        if device == "cuda":
            torch.cuda.synchronize()
        durations.append(h.time.perf_counter() - then)
        progress.completed("training_step", {"update": counters["updates"], "skips": counters["skips"]})
        total += value
        batches += 1
    h.need(batches > 0, "empty training loader")
    return dict(loss=total / batches, batches=batches, step_seconds=durations)


@torch.no_grad()
def evaluate(arm, model, loader, device, amp, full_corpus, crop_corpus, progress, *, tta=False, phase="validation"):
    """mil_screen_hooks_v3.evaluate_with with view ids and per-view flip TTA."""
    h.need(phase in ("validation", "panel_inference"), "invalid readout phase")
    progress.phase_start(phase)
    model.eval()
    predictions, ordered, batches = [], [], 0
    for x, _, ids, views in loader:
        with h.autocast(device, amp):
            logits = arm.forward(model, x.to(device), views.to(device))
        h.need(logits.shape == (len(ids), 12) and bool(torch.isfinite(logits).all()), "invalid evaluation logits")
        probabilities = torch.sigmoid(logits.float()).cpu().numpy()
        if tta:
            flipped = flip_views(x, views, ids, full_corpus, crop_corpus, loader.dataset.k)
            with h.autocast(device, amp):
                logits = arm.forward(model, flipped.to(device), views.to(device))
            h.need(logits.shape == (len(ids), 12) and bool(torch.isfinite(logits).all()), "invalid TTA logits")
            probabilities = (probabilities + torch.sigmoid(logits.float()).cpu().numpy()) * h.np.float32(.5)
        predictions.append(probabilities)
        ordered += ids
        batches += 1
        progress.completed(phase, {"batch": batches, "studies": len(ordered), "tta": tta})
    h.need(bool(predictions), "empty evaluation")
    return h.np.concatenate(predictions), ordered, batches
