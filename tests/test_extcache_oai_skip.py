"""B 2026-10-08: one undecodable OAI series is logged to failures.csv and skipped, never fatal."""
from tools.extcache import cli


def test_oai_bad_series_logged_and_skipped(tmp_path, monkeypatch):
    tasks = [(f"/raw/{k}.tar.gz", "S", k, "1", "R", "V00", "d") for k in ("a", "bad", "c")]
    added = []
    monkeypatch.setattr(cli, "require_space", lambda root: None)
    monkeypatch.setattr(cli, "oai_tasks", lambda root, visit, limit: list(tasks))
    monkeypatch.setattr(cli, "_done", lambda source, key, root: False)
    monkeypatch.setattr(cli, "gen_dir", lambda source, root: tmp_path)
    monkeypatch.setattr(cli, "read_index", lambda source, root: added)

    def fake_read(task):
        if task[2] == "bad":
            raise ValueError("all input arrays must have the same shape")
        return task[2]
    monkeypatch.setattr(cli, "read_oai_task", fake_read)
    monkeypatch.setattr(cli, "add_series", lambda source, series, root: added.append(series))
    r = cli.generate("oai", tmp_path, workers=1)
    assert (r["built"], r["failed"]) == (2, 1) and added == ["a", "c"]
    lines = (tmp_path / "failures.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "series_key,archive,error" and lines[1].startswith("bad,/raw/bad.tar.gz,")
    added.clear()
    r = cli.generate("oai", tmp_path, workers=1)  # a logged failure is not retried on resume
    assert r["failed"] == 0 and "bad" not in added
