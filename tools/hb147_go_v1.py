"""H-B-147 preregistered decision gate v1.0.0, 2026-10-13.

Consumes paired per-seed per-ruler per-target AUC deltas AFTER the seven-lens
readout; does not open labels or replace multi_lens_readout_l4_v2.evaluate.
"""
import numpy as np


def decide(deltas, image_deltas, *, all_lenses_complete=False, original_harm_guard_pass=False,
           image_reads_eligible=False):
    d = np.asarray(deltas, dtype=float)
    if d.shape != (2, 4, 12) or not np.isfinite(d).all():
        return dict(status="HOLD", reason="two seeds, four rulers, twelve finite target deltas required")
    mean = d.mean(axis=0)
    weak, macro = mean[:, [3, 5, 6, 8]].mean(axis=1), mean.mean(axis=1)
    pass_rulers = (weak >= .002) & (macro >= .002)
    images = np.asarray(image_deltas, dtype=float)
    complete = all_lenses_complete and image_reads_eligible and images.shape == (2,) and np.isfinite(images).all()
    guards = bool(original_harm_guard_pass and (mean >= -.003).all())
    go = bool(complete and guards and pass_rulers.sum() >= 3 and (images >= 0).all())
    return dict(status="GO_CONFIRM_OTHER_FOLD" if go else "NO_GO" if complete else "HOLD",
                paired_seed_mean_weak=weak.tolist(), paired_seed_mean_macro=macro.tolist(),
                same_rulers_passing=int(pass_rulers.sum()), harm_guard_pass=guards,
                image_reads_complete=bool(complete))
