import os
from pathlib import Path

import pytest

from knee_workspace import Workspace, workspace_root


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.delenv("KNEE_WORKSPACE_ROOT", raising=False)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/workspace.toml").write_text(
        'schema_version = 1\nlegacy_roots = ["H:/old/project"]\n'
        '[paths]\ndata = "storage/competition"\nexperiments = "artifacts/experiments"\n',
        encoding="utf-8",
    )
    return Workspace.load(tmp_path)


def test_old_relative_uri_and_absolute_paths_match(ws):
    expected = ws.root / "artifacts/experiments/EXP1/summary.json"
    for value in ["experiments/EXP1/summary.json", "workspace://experiments/EXP1/summary.json",
                  "H:\\old\\project\\experiments\\EXP1\\summary.json", ".\\experiments\\EXP1/summary.json"]:
        assert ws.resolve(value) == expected


def test_root_compatibility_is_pathlike_and_only_maps_first_component(ws, monkeypatch):
    monkeypatch.setenv("KNEE_WORKSPACE_ROOT", str(ws.root))
    root = workspace_root()
    assert Path(root) == ws.root
    assert root / "data" / "experiments" == ws.root / "storage/competition/experiments"
    assert ws.uri(root / "data" / "train.csv") == "workspace://data/train.csv"


def test_path_resolution_does_not_depend_on_cwd(ws, monkeypatch, tmp_path):
    unrelated = tmp_path / "elsewhere"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    monkeypatch.setenv("KNEE_WORKSPACE_ROOT", str(ws.root))
    assert workspace_root() / "data" == ws.root / "storage/competition"


def test_traversal_is_rejected(ws):
    with pytest.raises(ValueError):
        ws.resolve("workspace://data/../../secret")


def test_local_path_override(ws):
    (ws.root / "config/local.toml").write_text('[paths]\ndata = "custom/input"\n')
    assert Workspace.load(ws.root).resolve("data/train.csv") == ws.root / "custom/input/train.csv"
