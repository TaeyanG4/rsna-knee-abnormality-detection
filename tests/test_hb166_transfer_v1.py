"""CPU tests v1.0.0 for the H-B-166 small transfer (no network: subprocess SSH is never reached)."""
import hashlib
import json
import subprocess
import sys

import pytest

from tools import hb166_build_v1 as b
from tools import hb166_transfer_v1 as t

REGISTERED = (b.WORK / "registration.json").exists()


def test_ssh_host_and_port_validation():
    assert t.ssh_base("root@50.175.95.210", 51626)[-1] == "root@50.175.95.210"
    assert "StrictHostKeyChecking=yes" in t.ssh_base("root@h", 22) and "BatchMode=yes" in t.ssh_base("root@h", 22)
    for host in ("-oProxyCommand=x", "root@h;rm", "a b"):
        with pytest.raises(ValueError):
            t.ssh_base(host, 22)
    with pytest.raises(ValueError):
        t.ssh_base("root@h", 0)


def run_verify(root, mode, manifest):
    return subprocess.run([sys.executable, "-c", t.VERIFY, str(root), mode], input=json.dumps(manifest).encode(),
                          capture_output=True)


def entry(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def test_remote_verifier_detects_missing_tamper_and_preexisting_new_files(tmp_path):
    (tmp_path / "req").mkdir()
    (tmp_path / "req" / "a.txt").write_bytes(b"alpha")
    manifest = dict(required={"req/a.txt": entry(b"alpha")}, files={"new/b.txt": entry(b"beta")})
    pre = run_verify(tmp_path, "preflight", manifest)
    assert pre.returncode == 0 and json.loads(pre.stdout)["status"] == "OK"
    assert json.loads(run_verify(tmp_path, "verify", manifest).stdout)["missing"] == ["new/b.txt"]
    (tmp_path / "new").mkdir()
    (tmp_path / "new" / "b.txt").write_bytes(b"BETA")
    out = run_verify(tmp_path, "preflight", manifest)
    assert out.returncode == 2 and json.loads(out.stdout)["different"] == ["new/b.txt"]
    (tmp_path / "new" / "b.txt").write_bytes(b"beta")
    (tmp_path / "req" / "a.txt").write_bytes(b"alphA")
    out = json.loads(run_verify(tmp_path, "verify", manifest).stdout)
    assert out["status"] == "HOLD" and out["different"] == ["req/a.txt"]
    bad = dict(required={"../escape.txt": entry(b"x")}, files={})
    assert run_verify(tmp_path, "preflight", bad).returncode != 0


def test_result_names_must_stay_in_stage(tmp_path):
    for name in ("tools/hb166_fit_v1.py", ".local/scratch/slot-b/hb166/pilot-v100/../x.json", "../outside"):
        with pytest.raises(ValueError):
            t.check_result_names({name: {"bytes": 1, "sha256": "0"}})


@pytest.mark.skipif(not REGISTERED, reason="stage not registered yet")
def test_new_files_are_only_hb166_and_required_never_sent():
    new = t.new_files()
    assert all("hb166" in p for p in new)
    assert not any(p.endswith((".npy", ".pt")) for p in new)
    required = t.required_files(set(new))
    assert not set(required) & set(new)
    assert any(p.endswith("control-s42/final.pt") for p in required)
    assert ".local/scratch/slot-b/hb155/crop-cache-v220/side_unknown.npy" in required


def test_plan_only_commands_never_touch_network(monkeypatch):
    monkeypatch.setattr(t.subprocess, "Popen", lambda *a, **k: pytest.fail("network used"))
    monkeypatch.setattr(t.subprocess, "run", lambda *a, **k: pytest.fail("network used"))
    assert t.pull("root@h", 22, "fits", execute=False)["status"] == "PLAN_ONLY"
    assert t.main(["preflight", "--host", "root@h", "--port", "22"]) == 0
