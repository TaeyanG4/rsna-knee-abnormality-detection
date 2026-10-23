"""CPU tests v1.0.0 for the H-B-166 supervisor wrapper (policy itself is the reviewed hb155 code)."""
import json

import pytest

from tools import hb166_watchdog_v1 as w

FIT = ["python", "-B", "-m", "tools.hb166_fit_v1", "fit", "--config", "c.json", "--out", "o", "--device", "cuda"]
SMOKE = ["python", "-B", "-m", "tools.hb166_fit_v1", "smoke", "--config", "c.json", "--out", "o", "--device", "cuda"]
PANEL = ["python", "-B", "-m", "tools.hb166_panel_v1", "fit", "--seed", "42", "--out", "o", "--device", "cuda"]


def test_classify_allowlist():
    assert w.classify(FIT) == ("tools.hb166_fit_v1", "fit", "cuda")
    assert w.classify(PANEL) == ("tools.hb166_panel_v1", "fit", "cuda")
    for bad in (["python", "-B", "-m", "tools.hb155_fit_v1", "fit", "--device", "cuda"],
                ["python", "-m", "tools.hb166_fit_v1", "fit", "--device", "cuda"],
                ["python", "-B", "-m", "tools.hb166_fit_v1", "train", "--device", "cuda"],
                ["python", "-B", "-m", "tools.hb166_fit_v1", "fit", "--device"]):
        with pytest.raises(ValueError):
            w.classify(bad)


def run(tmp_path, *extra, attempt=None):
    attempt = attempt or (w.WORK / "attempts" / "pytest-never-created")
    return w.main(["--attempt", str(attempt), "--budget", str(w.WORK / "pytest-budget.json"), *extra])


def test_paths_must_stay_in_stage(tmp_path):
    with pytest.raises(SystemExit):
        w.main(["--attempt", str(tmp_path / "a"), "--budget", str(tmp_path / "b.json"), "--measurement", "--", *SMOKE])


def test_measurement_only_for_cuda_smokes(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, "--measurement", "--", *FIT)
    with pytest.raises(SystemExit):
        run(tmp_path, "--measurement", "--", *SMOKE[:-1], "cpu")


@pytest.mark.parametrize("command,rates", [(FIT, {"train_step": 1.}), (FIT, None), (PANEL, {"train_step": 1., "validation": 1.}),
                                           (SMOKE, {"train_step": 1., "validation": 1.})])
def test_scientific_runs_need_matching_measured_rates(tmp_path, command, rates):
    extra = []
    if rates is not None:
        path = tmp_path / "rates.json"
        path.write_text(json.dumps(rates))
        extra = ["--healthy-rates", str(path)]
    with pytest.raises(SystemExit):
        run(tmp_path, *extra, "--", *command)
    assert not (w.WORK / "attempts" / "pytest-never-created").exists()
    assert not (w.WORK / "pytest-budget.json").exists()
