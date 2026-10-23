# H224 Vast f3 after all-data — launch card v1.0.0

Changelog v1.0.0: additive f3 META_EPS entrypoint; explicit all-data run binding;
same GPU0 owner and single-attempt sequencing. Builder: astra-b2 / Codex.
Independent review: PENDING. This card does not authorize builder GPU execution.

## Schedule and scope

If adopted by B, Vast GPU0 runs **existing f2 -> existing all-data -> f3**.
The local GPU finishes its existing f1; its f3 waiter must be stopped and
verified **before f1 completion (~03:50 October 21)**. The existing f2 PID 3885
and `/root/work/wait_f2_then_all_v1.sh` are never modified, signalled or replaced.
No new waiter is installed by this build.

User-supplied measurements: Vast f2 0.55 s/update (226 windows/s), local
4070 3.1 s/update (41 windows/s; G: HDD cache). For f3's 15,690 updates,
the training-only projections are 2.40 h versus 13.51 h. Supplied calendar
estimates: all-data ends ~05:30, Vast f3 ~08:00, local f3 ~17:00 October 21.
These are estimates, not observed completion, launch conditions or timeouts.
The recipe, batch, optimizer, ten epochs, own holdout and frozen trainer stay
unchanged. Cross-host comparisons remain EXPOSED_EXPLORATORY.

`tools/l1_rco_k1_meta_f3_v1.py` is separate from the live
`tools/l1_rco_k1_meta_admit_v1.py`. **Do not replace the latter**: its hash is
bound into the existing f2/all reviews and its argv identifies all-data.
The new f3 command uses the original fold admission, plus freshly revalidated
META_EPS proof and a separate `f3/REVIEW-META-EPS.json` that binds the new code.
The old f2/all CLI continues to reject fold 3.

## Offline package and exact copy list

Builder artifact directory:
`.local/scratch/astra-b2/h224-vast-f3-v1/`

- Archive: `h224-vast-f3-v1.tar`
- Archive digest: `h224-vast-f3-v1.tar.sha256`
- Member digests: `F3-SHA256SUMS`
- Verification result: `PACKAGE-RESULT.json`

Archive payload/copy allowlist, relative to the existing project root:

```text
tools/l1_rco_k1_meta_f3_v1.py
tools/test_l1_rco_k1_meta_f3_v1.py
tools/l1_rco_k1_meta_f3_v1.md
tools/l1_rco_k1_device_plan_v1.md
F3-PREREQUISITES-SHA256SUMS
F3-SHA256SUMS
```

No existing executable, registration, review, parity receipt, admission, model,
label, prediction, job state, cache payload or waiter is included. The changed
device-plan document has backup
`.local/backups/astra-b2/h224-vast-f3-v1/l1_rco_k1_device_plan_v1.md`.
All code files in this build are new; the frozen source files are unchanged.
Transfer is B's action; the builder made no network connection.

On Vast, run in the **same project root and activated Python environment as
the existing f2/all commands**. Do not use a new checkout or change interpreter
spelling/symlinks: the exact resolved Python argv is part of the all-data binding.
Stage outside live fit directories, but within the project:

```bash
set -euo pipefail
task=.local/scratch/astra-b2/h224-vast-f3-v1
mkdir -p "$task/stage"
# B has placed the tar and its .sha256 sidecar in $task.
(cd "$task" && sha256sum -c h224-vast-f3-v1.tar.sha256)
tar -tf "$task/h224-vast-f3-v1.tar"
tar -xf "$task/h224-vast-f3-v1.tar" -C "$task/stage"
(cd "$task/stage" && sha256sum -c F3-SHA256SUMS)
# Existing executable dependencies must match; never overwrite them to fix a mismatch.
sha256sum -c "$task/stage/F3-PREREQUISITES-SHA256SUMS"
cp "$task/stage/tools/l1_rco_k1_meta_f3_v1.py" tools/
cp "$task/stage/tools/test_l1_rco_k1_meta_f3_v1.py" tools/
cp "$task/stage/tools/l1_rco_k1_meta_f3_v1.md" tools/
cp "$task/stage/tools/l1_rco_k1_device_plan_v1.md" tools/
cp "$task/stage/F3-PREREQUISITES-SHA256SUMS" "$task/"
cp "$task/stage/F3-SHA256SUMS" "$task/"
```

The archive builder checks exact member names, regular-file types and hashes.
Do not extract another archive over the project or copy the Windows CPU replay.
Preserve an existing different Linux device-plan document before replacing it.
If a deployed f3 file already differs or f3 has attempted admission/dispatch,
stop: do not overwrite, remove ledgers or turn this into a retry.

## Linux f3 CPU preparation and independent review

These are B/reviewer commands for the actual Vast filesystem. They do not
launch CUDA, read f2/all fit outputs or alter their receipts. To avoid competing
with live cache I/O, perform the full real CPU smoke after all-data terminates
if necessary. Preparation does not waive the independent pre-execution review.

```bash
cache=.local/scratch/astra-b2/h224-folds-v1/cache
task=.local/scratch/astra-b2/h224-vast-f3-v1
f3=.local/scratch/astra-b2/h224-folds-v1/f3
python -B tools/l1_rco_k1_meta_f3_v1.py register --fold 3 --cache "$cache"
python -B tools/l1_rco_k1_meta_f3_v1.py prepare --fold 3 --cache "$cache"
python -B tools/l1_rco_k1_meta_f3_v1.py validate --fold 3 --cache "$cache"
python -B tools/l1_rco_k1_meta_f3_v1.py cpu-smoke --fold 3 --cache "$cache"
python -B tools/l1_rco_k1_meta_f3_v1.py review-binding --fold 3 --cache "$cache" > "$task/linux-f3-bindings.json"
python -B tools/l1_rco_k1_meta_f3_v1.py admit --fold 3 --cache "$cache" --dry-run
```

Expected: train 3138, holdout 1029; smoke PASS with CUDA uninitialized.
`review-binding` revalidates the existing full-cache META_EPS receipt; it does
not rewrite it. A mismatch is HOLD. Before the independent review file exists,
`admit` must report BLOCKED. CPU CLI JSON status must be checked; exit zero on a
dry-run alone does not imply admission.

An actual independent reviewer must review the diff, Linux registration,
prepared trainer/targets hashes, Linux CPU-smoke receipt, existing META_EPS
provenance and launch policy. The reviewer writes a **new** file:

`.local/scratch/astra-b2/h224-folds-v1/f3/REVIEW-META-EPS.json`

Required fields:

```text
verdict: PASS (only if the independent reviewer actually passes it)
builder: astra-b2
reviewer: actual distinct reviewer identity
report: project-relative English report path
report_sha256: SHA256 of that actual report
bindings: exact bindings object from linux-f3-bindings.json
meta_eps_bindings: exact meta_eps_bindings object from linux-f3-bindings.json
```

Do not copy local f3 `REVIEW.json`, f2/all reviews or Windows smoke evidence;
do not fabricate PASS. Re-preparing/re-smoking changes the bound hashes and
requires refreshed review. The file binds the original fold contract, fullcache
registration, exact prepared artifacts, smoke, parity receipt, local32 reference,
META NPZ and old/new admission/sequence/runtime code.

```bash
# After independent PASS: require status CPU_ADMISSIBLE and blockers [].
python -B tools/l1_rco_k1_meta_f3_v1.py admit --fold 3 --cache "$cache" --dry-run
```

## Stop the local waiter safely before adopting Vast f3

**B only; these commands were not executed by the builder.** Stop the exact
waiter process, not f1, its training child, all Python processes or a process
tree. The following PowerShell command uses psutil PID/start-time validation,
requires the waiter to be WAITING and f1 to still be running, then verifies
absence of local f3 dispatch/admission. It preserves all files.

```powershell
Set-Location 'H:\kaggle\competitions\rsna-knee-abnormality-detection'
$py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
@'
from pathlib import Path
import psutil
from tools import l1_rco_k1_localwait_v1 as w
root = Path.cwd().resolve()
target = (root / "tools/l1_rco_k1_localwait_v1.py").resolve()
spec = w.plan()
ledger = Path(spec["waiter_ledger"])
def no_dispatch():
    assert not Path(spec["ledger"]).exists(), "HOLD: local f3 dispatch already attempted"
    assert not (root / "artifacts/logs/H224_K1_V3G_F3").exists(), "HOLD: local f3 job exists"
    assert not (w.k.BASE / "f3/fit/admission.json").exists(), "HOLD: f3 admission exists"
matches = []
for p in psutil.process_iter(["pid", "name"]):
    if "python" not in (p.info["name"] or "").lower():
        continue
    try:
        argv = p.cmdline()
        if any("l1_rco_k1_localwait_v1.py" in arg for arg in argv):
            if any(Path(arg).resolve() == target for arg in argv if arg.endswith(".py")):
                assert "launch-after-f1" in argv, "HOLD: unexpected waiter action"
                assert Path(p.cwd()).resolve() == root, "HOLD: unexpected waiter cwd"
                matches.append((p, p.create_time(), argv))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue
assert len(matches) == 1, "HOLD: require exactly one identified waiter"
p, started, argv = matches[0]
assert w.k.read(ledger)["state"] == "WAITING", "HOLD: waiter is not WAITING"
assert w.seq.runtime.job_status(root / "artifacts/logs", "H224_K1_V3G_F1")["state"] == "running", "HOLD: f1 is no longer running"
no_dispatch()
assert p.create_time() == started and p.cmdline() == argv, "HOLD: process identity changed"
p.terminate()  # exact waiter only; psutil protects against PID reuse
p.wait(timeout=10)
assert not p.is_running(), "HOLD: waiter is still alive"
no_dispatch()
assert w.k.read(ledger)["state"] in ("WAITING", "STOPPED"), "HOLD: dispatch race; investigate"
print("LOCAL_WAITER_STOPPED_NO_F3_DISPATCH", p.pid)
'@ | & $py -B -
if ($LASTEXITCODE -ne 0) { throw 'HOLD: do not adopt Vast f3 until resolved' }
```

A terminated waiter that never dispatched cannot wake at f1 SUCCESS and launch
f3. Hard termination may leave its immutable ledger saying WAITING; this is
historical state, not a live process. Do not delete the ledger or restart it.
The unchanged waiter's KeyboardInterrupt path is also regression-tested:
STOPPED, zero dispatches. **Stopping after dispatch does not cancel an already
launched f3.** If any assertion fails or completion races the stop, HOLD the
Vast switch and inspect the exact owned job; do not blindly kill descendants.

## Bound all-data SUCCESS, then one f3 launch

There is no time-based launch and no new waiter. B runs the next commands after
all-data has terminated, with Linux independent review complete and local waiter
shutdown verified. `launch-after-all` returns BLOCKED if all-data is running,
absent, stale, failed, legacy/unbound, wrong run/command/owner or still has a live
child. It accepts only the original META_EPS all-data argv in the GPU0 owner.

```bash
# Read only the all-data identity metadata, not its fit outputs.
all_run_id="$(python -B -c 'from tools import l1_rco_k1_meta_f3_v1 as w; p=w.k.BASE/"gpu0-owner/artifacts/logs/H224_K1_V3G_ALL/job.json"; print(w.k.read(p)["run_id"])')"
python -B tools/l1_rco_k1_meta_f3_v1.py launch-after-all --fold 3 --all-run-id "$all_run_id" --dry-run
# Require PLAN_ONLY, blockers [], actual independent PASS, local stop confirmed.
# B ONLY: one attempt; do not retry a failed/interrupted launch.
python -B tools/l1_rco_k1_meta_f3_v1.py launch-after-all --fold 3 --all-run-id "$all_run_id"
```

Owner: `.local/scratch/astra-b2/h224-folds-v1/gpu0-owner`.
Tag: `H224_K1_V3G_F3`.
Durable attempt ledger: `.local/scratch/astra-b2/h224-folds-v1/after-all-f3-v1.json`.
Admission: `h224-folds-v1/f3/fit/admission.json`.
The existing sequence lock, occupied-owner check and gpu_job atomic GPU claim
prevent two GPU jobs. The run child repeats exact-artifact review/cache admission
before training. No direct `run`, old `launch-pair`, old `launch-sequential`,
local waiter restart or receipt/ledger deletion is part of this card.
Failure/interruption preserves a STOPPED attempt; no automatic recovery.

## Builder evidence and limits

See `.local/scratch/astra-b2/h224-vast-f3-v1/BUILD-RESULTS.md`.
The builder executes the CPU commands through the same entrypoint in isolated
Windows scratch, plus synthetic bound-success dispatch tests. The real CPU smoke
uses the production cache and full Net backward/serving, with CUDA disabled.
The existing real32 META-noise panel exercises binding replay only; it is
TEST_ONLY and is never packaged as a production parity receipt.
No Linux host access, actual all-data SUCCESS, Linux f3 admission or independent
review PASS is claimed. Those steps remain required on Vast before launch.
