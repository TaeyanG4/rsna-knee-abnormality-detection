"""Synthetic CPU contracts for the reviewed adapter; no public source execution."""
import io
import os
import pickle

import numpy as np
import pytest
import torch

from training.adapters.goodpjw_reader import (
    ArraySeriesLoader, DecodedSlice, KneeNet, LABELS, NpySeriesLoader, SIZE,
    StudyDataset, build_study_table, canonical_resample, make_views,
    preprocess_decoded_series, slot_of, window_centres,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_one_thread():
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        pytest.fail("Run with CUDA_VISIBLE_DEVICES='' or '-1' explicitly")
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def row(series, plane="Sagittal", fat=1, study="s"):
    return {"StudyInstanceUID": study, "SeriesInstanceUID": series,
            "Anatomical_Plane": plane, "Fat_Suppression": fat}


def cache(value, slices=8):
    return np.full((slices, SIZE, SIZE), value, dtype=np.uint8)


def test_slot_mapping_and_stable_retained_count_order():
    assert [slot_of(p, f) for f in (1, 0) for p in ("Sagittal", "Coronal", "Axial")] == list(range(6))
    rows = [row("a"), row("z"), row("b"), row("short"), row("ax", "Axial", 0)]
    counts = {"a": 3, "z": 8, "b": 8, "short": 2, "ax": 4}
    table = build_study_table(rows, counts)
    assert table["s"][0] == ["z", "b", "a"]  # equal counts retain CSV order
    assert table["s"][5] == ["ax"]
    assert all(not table["s"][s] for s in (1, 2, 3, 4))
    with pytest.raises(ValueError, match="Duplicate"):
        build_study_table([row("a"), row("a")], counts)
    for bad in (float("nan"), 2, "unknown"):
        with pytest.raises(ValueError):
            slot_of("Sagittal", bad)
    with pytest.raises(ValueError):
        build_study_table([row("a")], {"a": float("nan")})


def test_series_above_retention_cap_tie_in_source_csv_order():
    rows = [row("first_65"), row("later_100")]
    decoded = {"first_65": 65, "later_100": 100}
    retained = {uid: min(n, 64) for uid, n in decoded.items()}
    # The reviewed runner passes retained n, matching source train.py/knee.py.
    assert build_study_table(rows, retained)["s"][0] == ["first_65", "later_100"]

def test_window_bounds_and_stratified_sampling():
    np.testing.assert_array_equal(window_centres(3, 12, False), np.ones(12, dtype=int))
    np.testing.assert_array_equal(window_centres(64, 12, False), [3, 8, 13, 18, 24, 29, 34, 39, 45, 50, 55, 60])
    samples = window_centres(64, 8, True, rng=np.random.default_rng(9))
    edges = np.linspace(.04 * 63, .96 * 63, 9)
    assert np.all(samples >= edges[:-1] - .5) and np.all(samples <= edges[1:] + .5)
    assert np.all(np.diff(samples) >= 0)
    for n, k in ((2, 12), (4, 0), (float("nan"), 1), (4, 1.5)):
        with pytest.raises(ValueError):
            window_centres(n, k, False)


def test_canonical_identity_transpose_and_flip():
    image = np.arange(SIZE * SIZE, dtype=np.float32).reshape(SIZE, SIZE)
    coronal = (1, 0, 0, 0, 0, -1)
    np.testing.assert_array_equal(canonical_resample(image, (.4, .4), coronal, 1), image)
    np.testing.assert_array_equal(canonical_resample(image.T, (.4, .4), (0, 0, -1, 1, 0, 0), 1), image)
    np.testing.assert_array_equal(canonical_resample(image[:, ::-1], (.4, .4), (-1, 0, 0, 0, 0, -1), 1), image)
    np.testing.assert_array_equal(image, np.arange(SIZE * SIZE, dtype=np.float32).reshape(SIZE, SIZE))


def test_anisotropic_geometry_pads_without_square_stretch():
    image = np.ones((100, 150), dtype=np.float32)
    result = canonical_resample(image, (.4, .8))
    expected = np.zeros((384, 384), dtype=np.float32)
    expected[142:242, 42:342] = 1
    np.testing.assert_array_equal(result, expected)


def test_series_position_order_uint8_and_cap():
    orientation = (0, 1, 0, 0, 0, -1)  # sagittal; increasing LPS x after canonical sign
    frames = [DecodedSlice(np.full((384, 384), 10 * (x + 1), np.float32), (.4, .4),
                           orientation, (x, 0, 0), 9-x, str(x)) for x in (2, 0, 1)]
    result, info = preprocess_decoded_series(frames)
    assert result.dtype == np.uint8 and result.shape == (3, 384, 384)
    np.testing.assert_array_equal(result[:, 100, 100], [0, 128, 255])
    assert info["plane"] == "Sagittal" and info["slice_gap"] == 1
    frames = [DecodedSlice(np.full((4, 4), i, np.float32), (38.4, 38.4),
                           instance_number=i, source_id=str(i)) for i in range(65)]
    result, info = preprocess_decoded_series(frames, "Axial")
    assert result.shape == (64, 384, 384) and info["n_raw"] == 65 and info["n"] == 64
    assert result[0, 0, 0] == 0 and result[-1, 0, 0] == 255


@pytest.mark.parametrize("spacing", [(0, .4), (-1, .4), (float("nan"), .4), (float("inf"), .4), None])
def test_invalid_spacing_fails(spacing):
    with pytest.raises(ValueError, match="spacing"):
        canonical_resample(np.ones((8, 8), np.float32), spacing)


def test_nonfinite_empty_and_malformed_decoded_frames_fail():
    with pytest.raises(ValueError, match="finite"):
        canonical_resample(np.full((8, 8), np.nan), (.4, .4))
    with pytest.raises(ValueError, match="three"):
        preprocess_decoded_series([])
    with pytest.raises(ValueError, match="collinear"):
        canonical_resample(np.ones((8, 8)), (.4, .4), (1, 0, 0, 1, 0, 0))
    frame = DecodedSlice(np.ones((8, 8)), (.4, .4), rescale_slope=1)
    with pytest.raises(ValueError, match="both rescale"):
        preprocess_decoded_series([frame] * 3)


def test_dataset_masks_triplets_pickle_and_random_series_choice():
    table = build_study_table([row("a"), row("b"), row("ax", "Axial", 0)], {"a": 8, "b": 3, "ax": 5})
    volumes = {"a": np.broadcast_to(np.arange(8, dtype=np.uint8)[:, None, None], (8, 384, 384)),
               "b": cache(200, 3), "ax": cache(50, 5)}
    dataset = StudyDataset(["s"], table, targets=np.full((1, 12), .2), k=2, loader=ArraySeriesLoader(volumes))
    restored = pickle.loads(pickle.dumps(dataset))
    x, mask, pos, y = restored[0]
    assert x.shape == (6, 2, 3, 384, 384) and x.dtype == torch.uint8
    assert mask.tolist() == [True, False, False, False, False, True]
    assert pos[1:5].count_nonzero() == 0 and x[1:5].count_nonzero() == 0
    assert x[0, 0, :, 0, 0].tolist() == [0, 1, 2]
    assert x[0, 1, :, 0, 0].tolist() == [5, 6, 7]
    torch.testing.assert_close(y, torch.full((12,), .2))
    assert isinstance(pickle.loads(pickle.dumps(NpySeriesLoader("unused"))), NpySeriesLoader)
    train = StudyDataset(["s"], table, k=1, train=True, loader=ArraySeriesLoader(volumes))
    np.random.seed(7)
    selected = {int(train[0][0][0, 0, 1, 0, 0]) == 200 for _ in range(20)}
    assert selected == {False, True}


def test_dataset_bad_targets_missing_study_and_short_series_fail():
    loader = ArraySeriesLoader({"a": cache(1, 2)})
    for targets in (np.full((1, 12), np.nan), np.ones((1, 11)), np.full((1, 12), 2)):
        with pytest.raises(ValueError):
            StudyDataset(["s"], {}, targets=targets, loader=loader)
    with pytest.raises(ValueError, match="no valid"):
        StudyDataset(["missing"], {}, loader=loader)[0]
    with pytest.raises(ValueError, match="3..64"):
        StudyDataset(["s"], {"s": [["a"], [], [], [], [], []]}, loader=loader)[0]


def test_series_shared_views_and_normalization():
    image = torch.arange(384 * 384, dtype=torch.int64).remainder(255).to(torch.uint8).reshape(1, 1, 384, 384)
    x = image.expand(4, 3, 384, 384)
    torch.manual_seed(12)
    augmented = make_views(x, 32, True, group=2)
    torch.testing.assert_close(augmented[0], augmented[1], rtol=0, atol=0)
    torch.testing.assert_close(augmented[2], augmented[3], rtol=0, atol=0)
    assert not torch.equal(augmented[0], augmented[2])
    constant = make_views(torch.full((1, 3, 384, 384), 127, dtype=torch.uint8), 32, False, 1)
    expected = (torch.tensor(127 / 255) - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
    torch.testing.assert_close(constant[0, :, 16, 16], expected)


def test_offline_convnext_forward_gradient_mask_and_state_roundtrip():
    # Random initialization only. This does not read the pinned initialization or public weights.
    torch.manual_seed(2)
    model = KneeNet(pretrained=False, drop=0, drop_path=0).eval()
    x = torch.full((1, 6, 1, 3, 384, 384), 127, dtype=torch.uint8)
    mask = torch.tensor([[True, False, False, False, False, False]])
    pos = torch.full((1, 6, 1), .5)
    prediction = model(x, mask, pos, res=32)
    assert prediction.shape == (1, len(LABELS)) and torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    gradient = next(model.enc.parameters()).grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = KneeNet(pretrained=False, drop=0, drop_path=0).eval()
    restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
    x[:, 1:] = 255  # absent series must never affect the output
    with torch.no_grad():
        replay = restored(x, mask, pos, res=32)
    torch.testing.assert_close(replay, prediction.detach(), rtol=0, atol=0)
    with pytest.raises(ValueError, match="at least one"):
        model(x, torch.zeros_like(mask), pos, res=32)
    pos[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(x, mask, pos, res=32)
