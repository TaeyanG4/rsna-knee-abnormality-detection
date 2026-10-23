"""CPU-only, fake-data checks for the EXP130 builder and Gold58 scorer."""
import csv
import json

import pytest

from tools import exp130_gold58_compare as e


def fake_gold(root):
    path = root / "storage/competition/train.csv"
    path.parent.mkdir(parents=True)
    ids = [f"fake-study-{i:02d}" for i in range(58)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([e.UID, *e.TARGETS])
        for i, uid in enumerate(ids):
            writer.writerow([uid, *[int(i % (t + 3) == 0) for t in range(12)]])
        writer.writerow(["not-gold", "", *([""] * 11)])
    return ids


def test_official_gold_filters_only_labeled_rows(tmp_path):
    expected = fake_gold(tmp_path)
    ids, labels = e.official_gold(tmp_path)
    assert ids == expected
    assert len(labels) == 58
    assert all(len(row) == 12 for row in labels)


def test_auc_midrank_and_minimum_class_rule(tmp_path):
    assert e.auc([0, 0, 1, 1], [.1, .2, .8, .9]) == 1
    assert e.auc([0, 1, 0, 1], [.5, .5, .5, .5]) == .5
    assert e.auc([0, 0], [.1, .9]) is None
    ids = fake_gold(tmp_path)
    labels = e.official_gold(tmp_path)[1]
    vals = [[float(labels[i][t]) for t in range(12)] for i in range(58)]
    assert e.scores(labels, {"perfect": vals}, list(range(12)))["perfect"]["macro"] == 1
    assert len(ids) == 58


def test_read_predictions_refuses_bad_order_and_nonfinite(tmp_path):
    ids = fake_gold(tmp_path)
    values = [[.5] * 12 for _ in ids]
    path = tmp_path / "pred.csv"
    e.write_predictions(path, ids, values)
    assert e.prediction_csv(path, ids) == values
    with pytest.raises(ValueError, match="UID order"):
        e.prediction_csv(path, ids[::-1])
    bad = tmp_path / "bad.csv"
    values[0][0] = float("nan")
    e.write_predictions(bad, ids, values)
    with pytest.raises(ValueError, match="probabilities invalid"):
        e.prediction_csv(bad, ids)


def test_gold_kernel_retargets_only_main_body():
    notebook = json.loads((e.FIDELITY / "release.ipynb").read_text(encoding="utf-8"))
    original = "".join(notebook["cells"][6]["source"])
    changed = e.gold_kernel_source(original, [f"fake-{i}" for i in range(58)])
    assert changed[:changed.index("GOLD58_UIDS = ")] == original[:original.index("def main():\n")]
    assert 'gpu_windows(vol, mask, K_EVAL, res, d, norm=NORM, mirror=True)' in changed
    assert 'pd.read_csv(ROOT + "/train_series.csv", dtype=' in changed
    assert 'pd.read_csv(ROOT + "/test.csv")' not in changed
    assert 'out = "/kaggle/working/gold58_oai.csv"' in changed
    assert 'seconds_per_study=elapsed/N' in changed
    assert 'sub.head()' not in changed


def test_compare_fake_predictions_and_paired_bootstrap(tmp_path):
    ids = fake_gold(tmp_path)
    labels = e.official_gold(tmp_path)[1]
    local = tmp_path / "local"
    local.mkdir()
    perfect = [[.95 if labels[i][t] else .05 for t in range(12)] for i in range(58)]
    reverse = [[1-v for v in row] for row in perfect]
    oai = tmp_path / "oai.csv"
    e.write_predictions(oai, ids, reverse)
    for seed in (42, 43):
        for view in ("plain", "tta"):
            e.write_predictions(local / f"gold58_s{seed}_{view}.csv", ids, perfect)
    result = e.compare(oai, local=local, root=tmp_path, out=tmp_path / "report", draws=30)
    body = json.loads((tmp_path / "report/comparison.json").read_text(encoding="utf-8"))
    assert result["studies"] == 58
    assert body["models"]["radr50_s42"]["macro"] == 1
    assert body["models"]["oai_0949"]["macro"] == 0
    assert body["paired_study_bootstrap"]["draws"] == 30
    assert body["paired_study_bootstrap"]["differences"]["radr50_s42 - oai_0949"]["delta"] == 1
    assert "never a gate" in (tmp_path / "report/comparison.md").read_text(encoding="utf-8")
