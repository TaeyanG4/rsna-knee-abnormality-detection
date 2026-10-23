"""Resume a gpu_queue plan after a reboot or a killed run.

A job that was running when the machine went down has job.json but neither SUCCESS nor
FAILED, so gpu_queue.py would skip it as "already running". This clears those stale markers
(the partial experiment folder is kept under artifacts/experiments/<TAG>_partial_<time>)
and restarts the queue runner, which then skips finished jobs and reruns the rest.

  python tools/queue_resume.py artifacts/logs/queue-r0-384-20261002.json
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "src"))
from raptor_common import EXPERIMENTS, LOGS, ROOT  # noqa: E402


def alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:
        return False


def main(plan_path: str):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    cleared = []
    for job in plan:
        d = LOGS / job["tag"]
        if not (d / "job.json").exists() or (d / "SUCCESS").exists() or (d / "FAILED").exists():
            continue
        info = json.loads((d / "job.json").read_text(encoding="utf-8"))
        if info.get("pid") and alive(int(info["pid"])):
            print(f"{job['tag']}: still running (pid {info['pid']}), left alone"); continue
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (d / "job.json").rename(d / f"job.interrupted-{stamp}.json")
        exp = EXPERIMENTS / job["tag"]
        if exp.exists():
            exp.rename(EXPERIMENTS / f"{job['tag']}_partial_{stamp}")
        cleared.append(job["tag"])
    print("cleared interrupted:", cleared or "none")
    subprocess.run([sys.executable, str(ROOT / "tools" / "gpu_queue.py"), plan_path], cwd=ROOT, check=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
