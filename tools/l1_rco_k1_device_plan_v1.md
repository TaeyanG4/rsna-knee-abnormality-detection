# H224 device plan v1.2.0 — B launch card

Changelog v1.2.0: optional Vast f3 after all-data bound SUCCESS, using the
additive [Vast f3 launch card](l1_rco_k1_meta_f3_v1.md). If B adopts this route,
that card supersedes the **local f3 schedule and launch commands below**.
Stop and verify the existing local waiter before f1 finishes; do not run both
routes. Existing local registration/review/waiter and Vast f2/all files stay
untouched. New Linux f3 CPU qualification and independent META_EPS review are
required. No switch or GPU launch has been performed by the builder.

Changelog v1.1.0: explicit CPU `--k1-only` health gate without R7 inference.
Changelog v1.0.0: Vast runs **f2 only, then all-data**; local Windows runs **f3 after
the exact f1 run succeeds**. This replaces the old f2 -> f3 and dual-GPU launch
cards. Builder astra-b2; independent review PENDING. No GPU launch is implied.
Training sources, registrations, cache, recipe, owner scopes and run receipts
are unchanged. Never update a running render or a running training directory.

## Schedule and timing

| Device | Order | Training work | Planning time |
|---|---|---|---|
| Vast RTX4090 GPU0 | f2 -> all-data | 15,500 -> 20,830 updates | f2 about 8 h; all about 10.8 h |
| Local 4070 | existing f1 -> f3 | f3 15,690 updates | about 13–13.5 h plus inference |

The supplied schedule targets f2 at ~23:00 October 20, all-data after f2, and
local f3 after f1 SUCCESS (~04:00 October 21). Times are estimates, not launch
conditions or timeouts. All-data time = measured same-host f2 training time
times 20830/15500 = 1.3439, plus its final guard/export overhead. Eight-hour f2
implies 10.75 h. At the inherited local 3.1 s/update, f3 takes 13.51 h.
No running f1 timing/output was inspected by this builder.

All-data uses all 4,167 labelled studies (3190 + 977), the identical v3-G
optimizer/LR/warm-up/augmentation/batch recipe, ten fixed epochs and final EMA.
No own holdout or early checkpoint selection is possible: `clean_oof=false`.
Loss and final per-target nonconstant-prediction guards fail closed.
See `tools/l1_rco_k1_all_v1.md` for the unchanged scientific contract.

## Windows f3 preparation — CPU only

Run from `H:\kaggle\competitions\rsna-knee-abnormality-detection`.
These commands use only f3 work/output and the common cache. Do not prepare f1.

```powershell
$py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
$task = '.local/scratch/astra-b2/h224-folds-v1'
$device = '.local/scratch/astra-b2/h224-device-plan-v1'
$cache = 'G:/temp_data/rsna-knee/l1rco-cache-full'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = '1'
& $py -B tools/l1_rco_k1_folds_v1.py register --fold 3
& $py -B tools/l1_rco_k1_folds_v1.py prepare --fold 3 --cache $cache
& $py -B tools/l1_rco_k1_folds_v1.py validate --fold 3
& $py -B tools/l1_rco_k1_folds_v1.py cpu-smoke --fold 3 --cache $cache
& $py -B tools/l1_rco_k1_folds_v1.py evaluate --fold 3 --dry-run
& $py -B tools/l1_rco_k1_folds_v1.py admit --fold 3 --cache $cache --dry-run
& $py -B tools/l1_rco_k1_localwait_v1.py launch-after-f1 --dry-run
```

f3 has 3,138 training studies and its own 1,029-study holdout. The registration
is immutable; prepare refuses a consumed admission. CPU smoke exercises real
ConvNeXt backward/update, one real holdout study, spawned loader, exact resume
and forced collapse. CPU commands hide CUDA internally. The waiter's dry-run
is explicitly **plan only**: it does not read f1, inspect live jobs, or admit f3.

Independent review must produce `$task/f3/REVIEW.json` after reviewing exact
sources, generated trainer diff and these CPU receipts. Required contents:

```python
# Reviewer/B only after genuine independent PASS. Do not fabricate a verdict.
from tools import l1_rco_k1_folds_v1 as k
k.configure(3, 0)
reg = k.read(k.WORK / "registration.json")
# record = dict(verdict="PASS", builder="astra-b2", reviewer=ACTUAL_REVIEWER,
#               report=PROJECT_RELATIVE_PASS_REPORT,
#               report_sha256=k.sha(k.ROOT / PROJECT_RELATIVE_PASS_REPORT),
#               bindings=k.review_binding(reg))
# Write record to k.REVIEW with mode "x"; never overwrite an existing receipt.
```

`bindings` has exactly: contract_id, registration_sha256,
primary_cohort_id_list_sha256, prepare_sha256, cpu_smoke_sha256,
fullcache_registration_sha256. The report hash must match, and reviewer must
differ from builder. Repeat `admit --fold 3 --cache $cache --dry-run` after PASS;
it must say CPU_ADMISSIBLE. Re-preparing/re-smoking after review changes bindings.

## Local f3 launch — B only

B supplies the actual f1 run ID from B's own dispatch record. The following is
NOT a CPU check and was NOT executed by the builder:

```powershell
& $py -B tools/l1_rco_k1_localwait_v1.py launch-after-f1 --f1-run-id '<ACTUAL_F1_RUN_ID>'
```

Keep the foreground waiter alive in B's persistent terminal. It uses local
project ownership, not the Vast gpu0-owner directory. It binds the supplied f1
run ID, exact frozen command, cwd and command hash, waits for its modern bound
SUCCESS, and revalidates review/cache before dispatch. FAILED, stale, unknown,
legacy, replaced metadata or interruption stops it without launching f3.
One local GPU job is enforced again by gpu_job's atomic claim and the unchanged
training admission/memory gates. Another job winning that claim blocks f3.

After success it executes exactly:

```powershell
# Informational equivalent child dispatch; do NOT run this beside the waiter.
# Only B, after verified f1 SUCCESS, independent PASS and CPU_ADMISSIBLE.
& $py -B -c "import subprocess; from tools import l1_rco_k1_folds_v1 as k; subprocess.run(k.launch_command(fold=3,gpu=0), cwd=k.ROOT, check=True)"
```

The actual waiter supplies `KNEE_WORKSPACE_ROOT=project root` and
`CUDA_VISIBLE_DEVICES=0`, then awaits f3's bound terminal receipt.
Its durable `local-f3-waiter.json` and `local-f3-dispatch.json` are under
`$device`. Existing ledgers, f3 job directory or admission block retry.
Stopping the waiter never cancels a running fit. Preserve all evidence; B must
inspect any already-dispatched job before a separately reviewed recovery.

## Vast f2 only — same existing Linux project root

Keep the existing absolute project root and completed cache. Do not launch
from a newly extracted staging tree. Do not rerender or overwrite the live
render directory. Linux defaults to `$task/cache`; Windows cache paths must
not be copied into Linux prepare/review receipts.

After the existing render finishes, full-cache receipt and 32-study exact
H224-PARITY.json must pass. Existing platform qualification remains required:
runtime/software snapshot, unchanged batch, and serving-platform parity.
Historical renderer details remain in `tools/l1_rco_k1_folds_v1.md`; its
dual-GPU launch instructions are retired.

```bash
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
python -B tools/l1_rco_k1_folds_v1.py register --fold 2
python -B tools/l1_rco_k1_folds_v1.py prepare --fold 2
python -B tools/l1_rco_k1_folds_v1.py validate --fold 2
python -B tools/l1_rco_k1_folds_v1.py cpu-smoke --fold 2
python -B tools/l1_rco_k1_folds_v1.py admit --fold 2 --dry-run
python -B tools/l1_rco_k1_foldseq_v1.py launch-single --fold 2 --dry-run
# B only, after Linux-local independent REVIEW.json and CPU_ADMISSIBLE:
python -B tools/l1_rco_k1_foldseq_v1.py launch-single --fold 2
```

The launcher uses unchanged `k.launch_command(fold=2,gpu=0)`,
`$task/gpu0-owner`, canonical job `H224_K1_V3G_F2`, and
`$task/single-f2-v1.json`. It waits for bound SUCCESS and exits. It never queues
f3. CLI `launch-sequential` now refuses execution and `--fold 3` is invalid.
The historical `plan()` is retained only for regression compatibility.
Do not use old `launch-pair`, direct Vast f3 commands or an already-running old
sequencer. B resolves any old sequencer before installing the new launcher.

Prepare all-data on Linux using the CPU commands in
`tools/l1_rco_k1_all_v1.md` and obtain its distinct host-bound REVIEW.json.
After f2 SUCCESS and CPU_ADMISSIBLE, in the SAME root:

```bash
python -B tools/l1_rco_k1_allseq_v1.py launch-after-f2 --dry-run
# B only:
python -B tools/l1_rco_k1_allseq_v1.py launch-after-f2
```

allseq does not wait for f2: early invocation is BLOCKED. Do not place it in an
unconditional shell `;` chain after a failed f2 command. It rechecks f2's bound
receipt and owner; all-data uses that same owner, with no automatic retry.

## Export and private dataset preparation

Production runs export automatically. Keep every export's seven manifest
members plus EXPORT.json and SHA256SUMS together. B stages completed copies
locally at `$device/exports/f1`, `f2`, `f3`, `all` after their respective jobs
are terminal. Copy the declared export members only, without replacing local
registration/prepare/review or runtime job directories. f2/all come from Vast;
f1/f3 are local. Preserve producer `fit/ckpt/epochs.json` separately if later
producer `export` revalidation is needed.

The following are file-only commands; they never upload. Output must not
already exist: use verify-dataset on repeat, not package or deletion.

```powershell
foreach ($k in 1,2,3) {
    & $py -B -m tools.k1_fold_dataset_v1 verify-export --export "$device/exports/f$k/EXPORT.json"
    if ($LASTEXITCODE) { throw "Export f$k failed" }
    & $py -B -m tools.k1_fold_dataset_v1 package --export "$device/exports/f$k/EXPORT.json" --output "$device/dataset-f$k" --dataset-id "taeyangg4/knee-k1-v3gf$k-v1-assets"
    if ($LASTEXITCODE) { throw "Package f$k failed" }
    & $py -B -m tools.k1_fold_dataset_v1 verify-dataset --dataset "$device/dataset-f$k"
    if ($LASTEXITCODE) { throw "Verify f$k failed" }
}
& $py -B -m tools.k1_fold_dataset_v1 verify-export --export "$device/exports/all/EXPORT.json"
if ($LASTEXITCODE) { throw 'All-data export failed' }
& $py -B -m tools.k1_fold_dataset_v1 package --export "$device/exports/all/EXPORT.json" --output "$device/dataset-all" --dataset-id taeyangg4/knee-k1-v3gall-v1-assets
if ($LASTEXITCODE) { throw 'All-data package failed' }
& $py -B -m tools.k1_fold_dataset_v1 verify-dataset --dataset "$device/dataset-all"
```

Exact private dataset IDs: `taeyangg4/knee-k1-v3gf1-v1-assets`,
`taeyangg4/knee-k1-v3gf2-v1-assets`, `taeyangg4/knee-k1-v3gf3-v1-assets`,
`taeyangg4/knee-k1-v3gall-v1-assets`. Each package has one checkpoint and frozen
serving source, metadata `isPrivate=true`, no data/labels/predictions.
All-data additionally says own-holdout evaluation is impossible.

## Own-holdout gate before model use

### K1-only health gate (v1.1.0)

Use this explicit mode while R7 fk-only predictions for folds 1..3 are
unavailable and their GPU inference is not scheduled. It does not run or
establish the R7 complement comparison. The full comparison below remains
unchanged; K1 HEALTH_PASS is not release/submission authorization.

```powershell
& 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe' -B tools/l1_rco_k1_foldgate_v1.py --k1-only
```

The default inputs are f0's existing
`.local/scratch/l1-rco/k1v3-grouping/fit/{holdout.csv,result.json}` and
`$device/exports/fk/fit/{holdout.csv,result.json}` for k=1,2,3. Read completed
production export copies only; never point the gate at a running fit directory.
The default aggregate output is `$device/k1-only-health.json`, separate from
the full gate. No registration, prepare, admission or GPU command is needed.

For every available fold and each fixed ruler (jev_hybrid, aquila_v1, dread,
consensus), compute the unweighted macro AUC of all 12 targets on the exact
own holdout, retaining the existing ruler masks. HEALTH_PASS requires at
least one available fold in 1..3 and, for **every available k and every ruler**,
`K1_fk >= K1_f0 - 0.005`, with no invalid row or undefined target AUC.
This is one-sided: improvement above f0 is allowed. Numeric boundary
comparisons allow only 1e-12 rounding tolerance.

Folds with neither export file available are PENDING. A partial pair,
incorrect receipt/hash/count/contract, invalid probability/status/roster,
or undefined AUC is HOLD (exit 2), never silently omitted. f0 is mandatory.
With only f0, overall status is PENDING. With passing f1+f2 and missing f3,
overall status is HEALTH_PASS **for present folds**, f3 is PENDING, and
`all_folds_evaluated=false`. Re-run the same command after the f3 export arrives.
Inspect fold states and completion as well as overall status.

The same K1 production checks as the full gate apply: COMPLETE,
`cpu_test=false`, correct model fold, exact holdout row count and CSV SHA256,
and sealed v3-G seed0 contract. Output pins both input files and cohort hash,
records each ruler's macro and delta, and explicitly records
`r7_complement_comparison.status=NOT_RUN` with the reason. Results remain
EXPOSED_EXPLORATORY / clean_oof=false; cross-fold/host deltas are descriptive.

Optional `--manifest PATH` accepts schema `h224-k1-only-input/v1` with a
`folds` object whose keys are `"0"` and any subset of `"1","2","3"`.
Each entry has exactly `k1_csv` and `k1_result` paths (project-relative or
absolute). Omitted nonzero folds are PENDING. R7 fields are rejected.
`--k1-only --dry-run` reports reference coverage only; it is not a health pass.

### Full K1 / R7 complement comparison

Run the gate in the local full project, where the existing evaluation reference
tables are available. The portable training bundle carries evaluator code but
does not add these reference tables; no new reference-data transfer is needed.

Four **folds** means k=0,1,2,3; it does not mean four rulers. For each existing
fixed ruler (jev_hybrid, aquila_v1, dread, consensus), compute macro across all
12 targets on each exact own holdout: n=977,1094,1067,1029 respectively.
Compare K1-fk against R7-fk on the SAME roster and target/mask definitions.
Use R7's actual fk-only pre-stacker predictions, never a full-fold ensemble
or f0 predictions relabelled as fk.

The new gate applies the user's literal `within .005` as
`abs(K1_fk - K1_f0) <= .005` for k=1,2,3, and requires
`K1_fk >= R7_fk` on at least 3 of the 4 folds. It reports all four rulers and
conservatively requires every ruler to pass, avoiding post-result ruler choice.
This absolute band is stricter than fable's original one-sided "more than .005
below f0" falsifier; the independent pre-execution reviewer must review this
explicit interpretation. Do not silently change the policy after seeing results.
SCREEN_PASS_PENDING_REVIEW is not release authorization.

Create `$device/gate-input.json` with this schema and all four records:

```json
{
  "schema": "h224-own-holdout-input/v1",
  "folds": {
    "0": {
      "k1_csv": ".local/scratch/l1-rco/k1v3-grouping/fit/holdout.csv",
      "k1_result": ".local/scratch/l1-rco/k1v3-grouping/fit/result.json",
      "r7_csv": ".local/scratch/astra-b2/h224-device-plan-v1/r7/f0.csv",
      "r7_receipt": ".local/scratch/astra-b2/h224-device-plan-v1/r7/f0.json"
    },
    "1": {
      "k1_csv": ".local/scratch/astra-b2/h224-device-plan-v1/exports/f1/fit/holdout.csv",
      "k1_result": ".local/scratch/astra-b2/h224-device-plan-v1/exports/f1/fit/result.json",
      "r7_csv": ".local/scratch/astra-b2/h224-device-plan-v1/r7/f1.csv",
      "r7_receipt": ".local/scratch/astra-b2/h224-device-plan-v1/r7/f1.json"
    },
    "2": {
      "k1_csv": ".local/scratch/astra-b2/h224-device-plan-v1/exports/f2/fit/holdout.csv",
      "k1_result": ".local/scratch/astra-b2/h224-device-plan-v1/exports/f2/fit/result.json",
      "r7_csv": ".local/scratch/astra-b2/h224-device-plan-v1/r7/f2.csv",
      "r7_receipt": ".local/scratch/astra-b2/h224-device-plan-v1/r7/f2.json"
    },
    "3": {
      "k1_csv": ".local/scratch/astra-b2/h224-device-plan-v1/exports/f3/fit/holdout.csv",
      "k1_result": ".local/scratch/astra-b2/h224-device-plan-v1/exports/f3/fit/result.json",
      "r7_csv": ".local/scratch/astra-b2/h224-device-plan-v1/r7/f3.csv",
      "r7_receipt": ".local/scratch/astra-b2/h224-device-plan-v1/r7/f3.json"
    }
  }
}
```

Each R7 receipt must authenticate `csv_sha256`, `complete=true`,
`mode="screen_fk"`, `model_selection="fk_only"` (replace k with its digit),
`output_kind="pre_stacker"`, `clean_oof=false`. These are production evidence,
not a template to fabricate. K1 receipts must match the sealed v3-G seed0
contract, full holdout row count/hash and `cpu_test=false`.

```powershell
# CPU reference coverage only; does not inspect any f1 output.
& $py -B tools/l1_rco_k1_foldgate_v1.py --dry-run --output "$device/gate-coverage.json"
# B after all eight genuine prediction sets and their receipts exist:
& $py -B tools/l1_rco_k1_foldgate_v1.py --manifest "$device/gate-input.json"
```

Missing/invalid rows, undefined target AUC, wrong model/fold or incomplete
provenance is HOLD. Existing historical R7-f0 exact976 cannot be silently used
as full977: its one invalid study needs real corrected inference or a separately
reviewed versioned cohort decision. Four-ruler coverage is not model validation.
All output remains EXPOSED_EXPLORATORY / clean_oof=false. Cross-fold/host
comparisons are descriptive; K1 vs R7 is paired only within a common cohort.
All-data has no independent own holdout. Its use additionally requires the
fold gate, independent result review, serving qualification and the planned
S4 >= S3 ordering check. Scheduled refit is not evidence of gain.

## Portable package v4 and exact copy allowlist

```powershell
& $py -B tools/l1_rco_k1_foldpack_v4.py build
& $py -B tools/l1_rco_k1_foldpack_v4.py verify --root "$device/package-v4"
& $py -B tools/l1_rco_k1_foldpack_v4.py verify-archive
```

Transfer privately: `$device/h224-device-plan-v4.tar` and `.tar.sha256`.
Existing v2/v3 archives remain unchanged. This full portable bundle includes
derived labels and initialization/anatomy weights; no raw DICOM, full cache,
running job output, credentials or report text.
The builder needs the preserved v3 tar: k1n_config_v1.py is copied from its
hash-pinned v1.6.0 member, excluding unrelated concurrent member-weight work.
The shared workspace's newer configuration source is not modified.

On Vast, extract in a separate staging location and verify before copying:

```bash
sha256sum -c h224-device-plan-v4.tar.sha256
tar -xf h224-device-plan-v4.tar
python -B h224-device-plan-v4/tools/l1_rco_k1_foldpack_v4.py verify --root h224-device-plan-v4
```

From a verified **v3 installation**, copy exactly these changed/new paths
from staging into the existing absolute root (keep all other files):

```text
tools/l1_rco_k1_foldseq_v1.py
tools/test_l1_rco_k1_foldseq_v1.py
tools/exp156_k_sweep.py
tools/exp115_vlm_eval.py
tools/exp115_vlm_pilot.py
tools/l1_rco_k1_localwait_v1.py
tools/l1_rco_k1_foldgate_v1.py
tools/l1_rco_k1_foldpack_v4.py
tools/test_l1_rco_k1_device_plan_v1.py
tools/l1_rco_k1_device_plan_v1.md
tools/l1_rco_k1_all_v1.md
.local/scratch/astra-b2/h224-folds-v1/FOLDS.md
```

For an existing **v2 installation**, additionally copy these v3 additions:

```text
tools/l1_rco_k1_all_v1.py
tools/l1_rco_k1_allstate_v1.py
tools/l1_rco_k1_allsmoke_v1.py
tools/l1_k1all_trainer_v1.py
tools/l1_rco_k1_allseq_v1.py
tools/l1_rco_k1_foldpack_v3.py
tools/test_l1_rco_k1_all_v1.py
tools/k1_fold_dataset_v1.py
tools/k1n_config_v1.py
tools/c85r7_rco_overlay_v1.py
.local/scratch/astra-b1/k1-shared-render/dataset-k1-v3gs1/source/compact_train.py
```

Compare copied bytes against staging INVENTORY.json/SHA256SUMS. The staging
manifest covers the full bundle; do not overwrite an existing installation's
manifest to pretend it is a complete extracted tree. Never recursively overlay
the whole bundle on a live installation. No registration, cache, render, owner,
admission, review, job or fit directory appears in this copy allowlist.
Linux CPU preparation, exact cache parity and independent review are still
required on that host; Windows tests cannot establish Linux CUDA compatibility.
