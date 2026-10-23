"""H-B-166 small transfer v1.1.0 (2026-10-15): ship ONLY new H-B-166 files to the stopped-then-restarted
Vast 4090 that already holds native448, crop-cache-v220 and the H-B-155 stage; verify the rest in place.

Commands (default is a plan; nothing touches the network without --execute, which only B runs):
  inventory                      local: transfer-inventory.json = new files (sent) + required files (verified, never sent)
  preflight --host --port        remote read-only check of required files and of any pre-existing new file
  send --host --port [--execute] tar the new files over SSH (binary OS pipes), then verify every byte remotely
  results --what smoke|fits      on the 4090: <smoke|result>-inventory.json + file list (status JSON checked first)
  pull --what smoke|fits [--execute]  binary tar pull of that set, then local SHA256 verification
  send-review [--execute]        ship the independent reviewer's review.json, verify its hash remotely
No credentials are read; SSH uses BatchMode and StrictHostKeyChecking=yes (B's reviewed known_hosts).
A remote file that differs from the local pin is a HOLD: this tool never overwrites remote project files.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

from tools import hb166_fold3_build_v1 as b
from tools.hb150_io_v1 import atomic_json

VERSION = "1.1.0"
DEST = "/workspace/rsna-knee-abnormality-detection"
DEFAULT_HOST, DEFAULT_PORT = "root@50.175.95.210", 51626
NEW_CODE = ['tools/hb166_fold3_build_v1.py', 'tools/hb166_fold3_fit_v1.py', 'tools/hb166_fold3_watchdog_v1.py', 'tools/hb166_fold3_readout_v1.py', 'tools/hb166_fold3_transfer_v1.py', 'tests/test_hb166_fold3_v1.py']
STAGE_FILES = ["registration.json", "gold58-ids.json", "prepared.json", "cpu-prepared.json", "smoke-queue.json", "fit-queue.json", "fold2-result-review.md"]
VERIFY = r'''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve(); mode=sys.argv[2]; m=json.loads(sys.stdin.read())
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def safe(name):
    p=(root/name); r=p.resolve()
    assert r.is_relative_to(root) and r!=root, name
    assert not any(x.is_symlink() for x in (p,*p.parents) if x.is_relative_to(root) and x!=root), name
    return p
out={'mode':mode,'missing':[],'different':[],'present_new':[],'checked':0}
for name,info in m['required'].items():
    p=safe(name)
    if not p.is_file(): out['missing'].append(name); continue
    if p.stat().st_size!=info['bytes'] or digest(p)!=info['sha256']: out['different'].append(name)
    out['checked']+=1
for name,info in m['files'].items():
    p=safe(name)
    if not p.exists():
        if mode=='verify': out['missing'].append(name)
        continue
    if p.stat().st_size!=info['bytes'] or digest(p)!=info['sha256']: out['different'].append(name)
    else: out['present_new'].append(name)
    out['checked']+=1
out['status']='OK' if not out['different'] and not out['missing'] else 'HOLD'
print(json.dumps(out)); sys.exit(0 if out['status']=='OK' else 2)
'''


def entry(path):
    p = b.ROOT / path
    b.need(p.is_file() and not p.is_symlink(), "missing file: " + path)
    return dict(bytes=p.stat().st_size, sha256=b.sha(p))


def new_files():
    names = list(NEW_CODE)
    for name in STAGE_FILES:
        names.append(b.rel(b.WORK / name))
    for seed in b.SEEDS:
        for engineering in (False, True):
            names.append(b.rel(b.config_path(seed, engineering)))
        names.append(b.rel(b.WORK / "cpu" / b.member(seed) / "summary.json"))
    return sorted(set(names))


def required_files(new):
    """Verified in place on the 4090, never sent."""
    paths = {}
    for seed in b.SEEDS:
        c = b.read(b.config_path(seed))
        control = b.read(b.ROOT / c["control_config"])
        for p, digest in list(c["inputs"].items()) + list(control["inputs"].items()):
            if p not in new:
                paths[p] = digest
        paths.update(c["control_fits"])
    for directory in (b.CONTROL_CORPUS, b.CROP_CORPUS):
        for name in ("ids.npy", "masks.npy") + (("side_unknown.npy",) if directory == b.CROP_CORPUS else ()):
            paths[directory + "/" + name] = None
    for p in (b.CONTROL_WORK / "prepared.json", b.b155.WORK / "runtime-review.json",
              b.b155.WORK / "runtime" / "4090.json", b.ROOT / b.CROP_CORPUS / "qa-summary.json"):
        paths[b.rel(p)] = None
    out = {}
    for p, digest in sorted(paths.items()):
        info = entry(p)
        b.need(digest is None or info["sha256"] == digest, "local required file differs from its binding: " + p)
        out[p] = info
    return out


def inventory():
    new = new_files()
    files = {p: entry(p) for p in new}
    required = required_files(set(new))
    result = dict(schema="hb166-transfer/v1", version=VERSION, status="READY", destination=DEST, files=files,
                  required=required, bytes=sum(f["bytes"] for f in files.values()),
                  required_bytes=sum(f["bytes"] for f in required.values()),
                  note="files are sent; required files must already exist on the 4090 with identical bytes")
    atomic_json(b.WORK / "transfer-inventory.json", result)
    (b.WORK / "transfer-files.txt").write_text("".join(p + "\n" for p in new), encoding="utf-8", newline="\n")
    return dict(status="READY", files=len(files), bytes=result["bytes"], required=len(required),
                required_bytes=result["required_bytes"], inventory_sha256=b.sha(b.WORK / "transfer-inventory.json"))


def ssh_base(host, port):
    b.need(re.fullmatch(r"[A-Za-z0-9_.@-]+", host) is not None and not host.startswith("-"), "invalid SSH host")
    b.need(1 <= int(port) <= 65535, "invalid port")
    return ["ssh", "-p", str(int(port)), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", host]


def remote_check(host, port, mode, manifest=None):
    b.need(mode in ("preflight", "verify"), "invalid check mode")
    data = (manifest or (b.WORK / "transfer-inventory.json")).read_bytes()
    cmd = ssh_base(host, port) + ["python3 -c " + shlex.quote(VERIFY) + " " + shlex.quote(DEST) + " " + mode]
    proc = subprocess.run(cmd, input=data, capture_output=True)
    try:
        return json.loads(proc.stdout.decode().strip().splitlines()[-1])
    except (ValueError, IndexError):
        return dict(status="HOLD", reason="remote check failed", exit_code=proc.returncode,
                    stderr=proc.stderr.decode(errors="replace")[-2000:])


def pipeline(producer_cmd, consumer_cmd, producer_cwd=None, consumer_cwd=None):
    """Bytes stay in OS pipes (never PowerShell text conversion)."""
    producer = subprocess.Popen(producer_cmd, cwd=producer_cwd, stdout=subprocess.PIPE)
    consumer = None
    try:
        consumer = subprocess.Popen(consumer_cmd, cwd=consumer_cwd, stdin=producer.stdout)
        producer.stdout.close()
        remote, local = consumer.wait(), producer.wait()
        b.need(local == 0 and remote == 0, f"tar/SSH failed: {local}/{remote}; diagnose before rerunning")
    finally:
        for child in (consumer, producer):
            if child is not None and child.poll() is None:
                child.kill()
                child.wait()


def send(host, port, execute=False):
    m = b.read(b.WORK / "transfer-inventory.json")
    b.need(m.get("status") == "READY" and m.get("version") == VERSION, "inventory not ready")
    for name, info in m["files"].items():
        b.need(entry(name) == info, "local file changed after inventory: " + name)
    tar = ["tar", "-cf", "-", "-T", str(b.WORK / "transfer-files.txt")]
    receive = ssh_base(host, port) + ["mkdir -p " + shlex.quote(DEST) + " && tar -xf - -C " + shlex.quote(DEST)]
    plan = dict(status="PLAN_ONLY", files=len(m["files"]), bytes=m["bytes"], tar=tar, receive=receive)
    if not execute:
        return plan
    pre = remote_check(host, port, "preflight")
    b.need(pre.get("status") == "OK", "preflight HOLD: " + json.dumps(pre)[:2000])
    pipeline(tar, receive, producer_cwd=b.ROOT)
    post = remote_check(host, port, "verify")
    b.need(post.get("status") == "OK", "remote verification HOLD: " + json.dumps(post)[:2000])
    return dict(status="TRANSFER_VERIFIED", files=len(m["files"]), bytes=m["bytes"], verified=post["checked"])


def results(what="fits"):
    """Run on the 4090. smoke: CUDA smoke evidence for the reviewer; fits: final results for the readout.

    Status JSON is checked before any output is listed (never opens running outputs).
    """
    b.need(what in ("smoke", "fits"), "unknown result set")
    names = []
    if what == "fits":
        for seed in b.SEEDS:
            for d in (b.WORK / "fits" / b.member(seed),):
                b.need(b.read(d / "heartbeat.json").get("status") == "complete", "incomplete: " + b.rel(d))
        names += [b.WORK / n for n in ("registration.json", "prepared.json", "cpu-prepared.json", "review.json",
                                        "gpu-budget.json", "memory-admission.json", "fold2-result-review.md")]
        for seed in b.SEEDS:
            fit = b.WORK / "fits" / b.member(seed)
            names += [fit / n for n in ("heartbeat.json", "summary.json", "runtime.json", "final.pt",
                                        "val.csv", "val.receipt.json", "val_k24.csv", "val_k24.receipt.json",
                                        "val_k36.csv", "val_k36.receipt.json", "gold.csv", "gold.receipt.json")]
    for seed in b.SEEDS:
        for smoke in (b.member(seed),):
            d = b.WORK / "cuda" / smoke
            b.need(b.read(d / "heartbeat.json").get("status") == "complete", "smoke incomplete: " + b.rel(d))
            names += [d / n for n in ("heartbeat.json", "summary.json", "runtime.json", "healthy-rates.json") if (d / n).exists()]
    names += sorted((b.WORK / "attempts").glob("*/receipt.json"))
    files = {b.rel(p): entry(b.rel(p)) for p in names}
    stem = "result" if what == "fits" else "smoke"
    atomic_json(b.WORK / (stem + "-inventory.json"), dict(schema="hb166-results/v1", version=VERSION, what=what,
                                                          status="RESULTS_COMPLETE", files=files))
    (b.WORK / (stem + "-files.txt")).write_text("".join(p + "\n" for p in sorted(files)), encoding="utf-8", newline="\n")
    return dict(status="RESULTS_COMPLETE", what=what, files=len(files), bytes=sum(f["bytes"] for f in files.values()))


def check_result_names(files):
    root = b.WORK.resolve()
    for name in files:
        p = (b.ROOT / name).resolve()
        b.need(p.is_relative_to(root) and ".." not in Path(name).parts and "\n" not in name, "unsafe result path: " + name)
        if p.exists():
            b.need(entry(name) == files[name], "local result exists with different bytes: " + name)


def pull(host, port, what="fits", execute=False):
    stem = "result" if what == "fits" else "smoke"
    rel_inventory = b.rel(b.WORK / (stem + "-inventory.json"))
    fetch = ssh_base(host, port) + ["cat " + shlex.quote(DEST + "/" + rel_inventory)]
    stream = ssh_base(host, port) + ["tar -cf - -C " + shlex.quote(DEST) + " -T "
                                     + shlex.quote(DEST + "/" + b.rel(b.WORK / (stem + "-files.txt")))]
    extract = ["tar", "-xf", "-", "-C", str(b.ROOT)]
    if not execute:
        return dict(status="PLAN_ONLY", what=what, fetch=fetch, stream=stream, extract=extract)
    proc = subprocess.run(fetch, capture_output=True, check=True)
    m = json.loads(proc.stdout.decode())
    b.need(m.get("schema") == "hb166-results/v1" and m.get("status") == "RESULTS_COMPLETE" and m.get("what") == what,
           "remote result set incomplete")
    check_result_names(m["files"])
    pipeline(stream, extract, consumer_cwd=b.ROOT)
    for name, info in m["files"].items():
        b.need(entry(name) == info, "pulled result differs: " + name)
    atomic_json(b.WORK / (stem + "-inventory.json"), m)
    return dict(status="RESULTS_VERIFIED", what=what, files=len(m["files"]),
                bytes=sum(f["bytes"] for f in m["files"].values()))


def send_review(host, port, execute=False):
    """Ship the independent review and its measured memory/prior-review evidence."""
    review = b.WORK / "review.json"
    r = b.read(review)
    b.need(r.get("schema") == "hb166-review/v1" and r.get("verdict") == "PASS" and r.get("independent") is True,
           "review.json is not an independent PASS")
    b.need(b.admit(review)["status"] == "ADMITTED", "smoke-bound admission HOLD")
    name = b.rel(review)
    names = [name, b.rel(b.WORK / "memory-admission.json"), b.rel(b.WORK / "fold2-result-review.md")]
    manifest = json.dumps(dict(required={}, files={p: entry(p) for p in names})).encode()
    tar = ["tar", "-cf", "-", *names]
    receive = ssh_base(host, port) + ["tar -xf - -C " + shlex.quote(DEST)]
    if not execute:
        return dict(status="PLAN_ONLY", tar=tar, receive=receive)
    cmd = ssh_base(host, port) + ["python3 -c " + shlex.quote(VERIFY) + " " + shlex.quote(DEST) + " preflight"]
    pre = json.loads(subprocess.run(cmd, input=manifest, capture_output=True).stdout.decode().strip().splitlines()[-1])
    b.need(pre.get("status") == "OK", "remote review.json differs: " + json.dumps(pre))
    pipeline(tar, receive, producer_cwd=b.ROOT)
    cmd[-1] = cmd[-1].replace(" preflight", " verify")
    post = json.loads(subprocess.run(cmd, input=manifest, capture_output=True).stdout.decode().strip().splitlines()[-1])
    b.need(post.get("status") == "OK", "remote review verification HOLD")
    return dict(status="REVIEW_SENT", sha256=entry(name)["sha256"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("inventory", "preflight", "send", "results", "pull", "send-review"))
    p.add_argument("--what", choices=("smoke", "fits"), default="fits")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args(argv)
    try:
        if a.command == "inventory":
            result = inventory()
        elif a.command == "preflight":
            result = remote_check(a.host, a.port, "preflight") if a.execute else dict(
                status="PLAN_ONLY", command=ssh_base(a.host, a.port) + ["python3 -c <VERIFY> " + DEST + " preflight"])
        elif a.command == "send":
            result = send(a.host, a.port, a.execute)
        elif a.command == "results":
            result = results(a.what)
        elif a.command == "send-review":
            result = send_review(a.host, a.port, a.execute)
        else:
            result = pull(a.host, a.port, a.what, a.execute)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        result = dict(status="HOLD", reason=str(exc))
    print(json.dumps(result))
    return 2 if result.get("status") == "HOLD" else 0


if __name__ == "__main__":
    sys.exit(main())
