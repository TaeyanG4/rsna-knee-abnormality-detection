"""CPU-only contract tests; no third-party model code is executed."""
import csv
import json
import shutil

import pytest

from tools import exp130_gold58_compare as e130
from tools import exp131_gold58_convnext as e131


def fake_root(root):
    ids = [f"fake-study-{i:02d}" for i in range(58)]
    train = root / "storage/competition/train.csv"
    train.parent.mkdir(parents=True)
    with train.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([e130.UID, *e130.TARGETS])
        for i, uid in enumerate(ids):
            writer.writerow([uid, *[int((i + t) % 3 == 0) for t in range(12)]])
    nb = root / ".local/exp130-gold58/oai_gold58/release.ipynb"
    nb.parent.mkdir(parents=True)
    nb.write_text(json.dumps({"cells": [{"cell_type": "code", "source":
                                    ["GOLD58_UIDS = " + repr(tuple(ids)) + "\n"]}]}), encoding="utf-8")
    apex = root / e131.APEX.relative_to(e131.ROOT)
    apex.parent.mkdir(parents=True)
    shutil.copyfile(e131.APEX, apex)
    return ids


def test_builder_embeds_exp130_uids_and_safe_metadata(tmp_path):
    ids = fake_root(tmp_path)
    dest = tmp_path / "kernel"
    receipt = e131.build(root=tmp_path, dest=dest)
    nb = json.loads((dest / "release.ipynb").read_text(encoding="utf-8"))
    code = "".join(nb["cells"][1]["source"])
    meta = json.loads((dest / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert receipt["uids"] == 58
    assert all(uid in code for uid in ids)
    assert "module.run(data, 'train', ckpts, studies=json.loads(ids))" in code
    assert "spec_from_file_location('infer'" in code
    assert "pylibjpeg-openjpeg" in code
    assert "Apache 2.0" in "".join(nb["cells"][0]["source"])
    assert meta["is_private"] and meta["enable_gpu"] and not meta["enable_internet"]
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["dataset_sources"] == [e131.DATASET]
    assert meta["competition_sources"] == ["rsna-knee-abnormality-detection"]
    with pytest.raises(ValueError, match="exists"):
        e131.build(root=tmp_path, dest=dest)


def test_apex_weights_and_rank_blend():
    assert e131.check_apex() == e130.sha256(e131.APEX)
    a = [[.1 + i * .1] * 12 for i in range(4)]
    b = list(reversed(a))
    got = e131.weighted_rank(a, b)
    assert got[0][e130.TARGETS.index("Baker's")] == .25
    assert got[-1][e130.TARGETS.index("Fracture")] == 1.0
    assert e131.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)


def test_compare_fake_data_paired_study_contrasts(tmp_path):
    ids = fake_root(tmp_path)
    labels = e130.official_gold(tmp_path)[1]
    good = [[.95 if labels[i][t] else .05 for t in range(12)] for i in range(58)]
    bad = [[1 - v for v in row] for row in good]
    oai = tmp_path / "oai.csv"
    cnx = tmp_path / "cnx.csv"
    e130.write_predictions(oai, ids, bad)
    e130.write_predictions(cnx, ids, good)
    local = tmp_path / "local"
    local.mkdir()
    for seed in (42, 43):
        for view in ("plain", "tta"):
            e130.write_predictions(local / f"gold58_s{seed}_{view}.csv", ids, good)
    result = e131.compare(oai=oai, cnx=cnx, local=local, root=tmp_path,
                          out=tmp_path / "report", draws=20)
    body = json.loads((tmp_path / "report/comparison.json").read_text(encoding="utf-8"))
    assert result["studies"] == 58
    assert body["models"]["oai_0949"]["macro"] == 0
    assert body["models"]["convnext"]["macro"] == 1
    assert body["models"]["radr50_s42"]["macro"] == 1
    assert body["paired_study_bootstrap"]["draws"] == 20
    assert "B1_cnx_085_015 - B2_radr_090_010" in body["paired_study_bootstrap"]["differences"]
    assert body["partner_spearman_vs_oai"]["convnext"]["mean_eligible"] == pytest.approx(-1)
    assert "optimistic on Gold58" in (tmp_path / "report/comparison.md").read_text(encoding="utf-8")
