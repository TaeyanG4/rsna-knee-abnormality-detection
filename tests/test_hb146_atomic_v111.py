"""Windows sharing violations: durable receipts fail closed, telemetry cannot stop fits."""
import pytest
from tools import hb146_pl_labels_v111 as pl


def test_transient_replace_retry(tmp_path, monkeypatch):
    real = pl.os.replace
    attempts, delays = [], []
    def flaky(src, dest):
        attempts.append(1)
        if len(attempts) < 4:
            raise PermissionError("sharing violation")
        real(src, dest)
    monkeypatch.setattr(pl.os, "replace", flaky)
    monkeypatch.setattr(pl.time, "sleep", delays.append)
    path = tmp_path / "receipt.json"
    pl.write(path, {"complete": True})
    assert pl.read(path) == {"complete": True}
    assert len(attempts) == 4 and delays == [.05, .1, .05 * 3]


def test_exhausted_status_nonfatal_receipt_fatal(tmp_path, monkeypatch):
    attempts = []
    def locked(*args):
        attempts.append(1)
        raise PermissionError("sharing violation")
    monkeypatch.setattr(pl.os, "replace", locked)
    monkeypatch.setattr(pl.time, "sleep", lambda *_: None)
    assert pl.write_status(tmp_path / "heartbeat.json", {"status": "running"}) is False
    assert len(attempts) == 4
    with pytest.raises(PermissionError):
        pl.write(tmp_path / "receipt.json", {"complete": True})
    assert len(attempts) == 8


def test_status_open_failure_nonfatal(tmp_path, monkeypatch):
    def locked(*args, **kwargs):
        raise PermissionError("sharing violation")
    monkeypatch.setattr(pl.Path, "write_text", locked)
    assert pl.write_status(tmp_path / "heartbeat.json", {}) is False
