"""H-B-124 hooks CPU engineering checks v1.0.0; real images, synthetic targets.

No label/report access, CUDA, lifecycle PASS note or scientific score.
Run: python -B -m tools.hb124_hook_smoke_v1
"""
import hashlib
import json
import os
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch

from training.arms import mil_screen_hooks_v1 as hooks
from training.arms.hb124_series_arm_v1 import ARM, TRAIN, PANEL, BoundProvenance
from tools.hb124_hook_build_v1 import WORK, documents


def close(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()


def loader_for(spec, ids, k, train, arm, corpus_dir, *, batch=1, persistent=False):
    zeros = {uid: np.zeros(12, np.float32) for uid in ids}
    ds = hooks.MILWindows(corpus_dir, ids, zeros, k, train, spec.document["seed"])
    ds = arm.dataset(ds, spec, hooks.ROOT, train=train)
    return hooks.DataLoader(ds, batch_size=batch, shuffle=train, drop_last=train, num_workers=2,
        persistent_workers=persistent, worker_init_fn=hooks.worker_init,
        generator=torch.Generator().manual_seed(spec.document["seed"]), collate_fn=arm.collate_fn)


def loader_parity(seed):
    spec = documents(seed)[3]
    corpus = hooks.common.Corpus(hooks.ROOT / TRAIN)
    ids = [str(u) for u in corpus.ids[:17]]
    provenance = BoundProvenance(corpus, corpus.dir / "picks.jsonl")
    streams = []
    for arm in (hooks.IDENTITY, ARM):
        loader = loader_for(spec, ids, 12, True, arm, corpus.dir, batch=8, persistent=True)
        epochs = []
        try:
            for epoch in range(3):
                loader.dataset.set_epoch(epoch)
                digest = hashlib.sha256()
                for x, y, ordered, *extra in loader:
                    digest.update(x.numpy().tobytes())
                    digest.update(y.numpy().tobytes())
                    digest.update(json.dumps(ordered).encode())
                    if arm is ARM:
                        for i, uid in enumerate(ordered):
                            centers = hooks.common.select_centers(corpus.mask(uid), 64, 12, corpus.spec,
                                "within-slot-valid", hooks.common.study_rng(seed, epoch, uid))
                            expected = provenance.windows(uid, centers)
                            assert all(torch.equal(extra[0][key][i], expected[key]) for key in expected)
                epochs.append(digest.hexdigest())
        finally:
            close(loader)
        streams.append(epochs)
    assert streams[0] == streams[1] and len(set(streams[0])) == 3
    return dict(seed=seed, epochs=3, batch=8, workers=2, persistent_workers=True,
                batches_per_epoch=2, images_targets_ids_exact=True, exact_center_metadata=True,
                epoch_sha256=streams[0])


def smoke():
    start = time.perf_counter()
    spec = documents(42)[3]
    corpus = hooks.common.Corpus(hooks.ROOT / TRAIN)
    uid = str(corpus.ids[0])
    hooks.configure_runtime("none", "cpu", policy="strict")
    torch.random.default_generator.manual_seed(42)
    model = hooks.build_model(spec.document["config"]["model"], hooks.ROOT)
    before = {k: v for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    ARM.attach(model, spec, hooks.ROOT)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert set(model.state_dict()) - set(before) == {"series_context." + k for k in model.series_context.state_dict()}
    del before
    train = loader_for(spec, [uid], 12, True, ARM, corpus.dir, persistent=True)
    train.dataset.set_epoch(0)
    try:
        x, _, _, metadata = next(iter(train))
        model.eval()
        with torch.no_grad():
            assert torch.equal(model(x), ARM.forward(model, x, metadata))
            dummy = torch.zeros(1, 1, 3, 224, 224)
            assert torch.equal(model(dummy), ARM.forward(model, dummy, ARM.probe("cpu")))
        grads, projection = [], []
        def finite(grad):
            assert torch.isfinite(grad).all()
            grads.append(True)
        handles = [p.register_hook(finite) for p in model.parameters() if p.requires_grad]
        handles.append(model.series_context.out.weight.register_hook(lambda g: projection.append(float(g.abs().sum()))))
        opt, sched, scaler = hooks.optimizer_stack(model, 1, "cpu", "none")
        counters = dict(updates=0, skips=0)
        record = hooks.train_epoch_with(ARM, model, train, opt, sched, scaler, torch.ones(12), "cpu", "none", 0, counters)
        for handle in handles:
            handle.remove()
        assert len(grads) == sum(p.requires_grad for p in model.parameters()) and projection[0] > 0
        assert counters == dict(updates=1, skips=0)
        del opt, sched, scaler
    finally:
        close(train)
    WORK.mkdir(parents=True, exist_ok=True)
    checkpoint = WORK / "hook-engineering-model.pt"
    model.eval()
    with torch.no_grad():
        expected = ARM.forward(model, x, metadata)
    torch.save(model.state_dict(), checkpoint)
    with torch.no_grad():
        model.series_context.out.weight.zero_()
    hooks.strict_load(model, torch.load(checkpoint, map_location="cpu", weights_only=True))
    with torch.no_grad():
        assert torch.equal(expected, ARM.forward(model, x, metadata))
    paths = []
    for k, tta in ((24, False), (36, False), (36, True)):
        loader = loader_for(spec, [uid], k, False, ARM, corpus.dir)
        pred, ordered, _ = hooks.evaluate_with(ARM, model, loader, "cpu", "none", corpus, tta=tta)
        assert ordered == [uid] and pred.shape == (1, 12) and np.isfinite(pred).all()
        paths.append(dict(corpus="train", k=k, flip=tta))
    empty_ids = [str(u) for u in corpus.ids if not corpus.mask(u).any()]
    assert empty_ids
    loader = loader_for(spec, empty_ids[:1], 36, False, ARM, corpus.dir)
    x0, _, _, meta0 = next(iter(loader))
    with torch.no_grad():
        assert not meta0["valid"].any() and torch.equal(model(x0), ARM.forward(model, x0, meta0))
    panel = hooks.common.Corpus(hooks.ROOT / PANEL)
    pmeta = BoundProvenance(panel, panel.dir / "picks.jsonl", panel=True)
    # Every panel study audited by constructor, including copied Sag0 provenance.
    panel_uid = str(next(u for u in panel.ids if (panel.mask(u) > 0).all()))
    loader = loader_for(spec, [panel_uid], 36, False, ARM, panel.dir)
    pred, ordered, _ = hooks.evaluate_with(ARM, model, loader, "cpu", "none", panel, tta=True)
    assert ordered == [panel_uid] and np.isfinite(pred).all()
    paths.append(dict(corpus="external-panel", k=36, flip=True))
    assert not torch.cuda.is_initialized()
    return dict(status="LABEL_FREE_HOOK_ENGINEERING_PASS", real_images=True, targets="synthetic-zero",
        zero_init_exact=True, attach_preserves_control_state_rng=True, finite_gradients=len(grads),
        projection_gradient_positive=True, train=record, optimizer_steps=1, strict_full_model_reload=True,
        empty_bag_identity=True, paths=paths, panel_metadata_studies=len(pmeta.by_uid),
        panel_copied_series=pmeta.copies, cuda_initialized=False, gpu_hours=0,
        lifecycle_validation="not run: label access prohibited", seconds=time.perf_counter() - start)


def main():
    import cv2
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    result = dict(loader_parity=[loader_parity(s) for s in (42, 43)], smoke=smoke())
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "hook-engineering-v1.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
