# D-A-052 fullbag launch card v1.1.0

Builder: astra-b2 / Codex. Independent review: PENDING.
Changelog v1.1.0: repair offline prerequisites against device-plan-v4 +
META_EPS + f3 installation inventories; missing-only sealed parent copies;
preflight every collision before copying. Training code/contract unchanged.
Changelog v1.0.0: isolated fullbag fold0 mode, streamed encoder-gradient replay,
cap640, bound-f3 one-shot META_EPS admission. No existing trainer, fold
entrypoint, live sequence or waiter is replaced.

## Registered experiment

Use `tools/l1_rco_k1_fullbag_v1.py ACTION --mode fullbag`.
Preserve the sealed v3-G seed0 split: **3190 train / 977 holdout**. The request
also says 976; dropping a row would change the sealed experiment, so the
implementation deliberately requires 977. Preserve targets, initialization,
class attention, batch2, crop352, ten epochs, LR1e-4/head3e-4, warmup1595,
15950 updates, weight decay .02, clipping5, EMA .999 with the original warm
EMA schedule, crop augmentation, checkpointing and eval stride3.

The sole scientific change is the full stride-based candidate bag instead of
uniform cap64, with one original weighted study BCE. Retain all candidates;
cap640 is a fail-closed bound, never a sampler. A metadata-only census of all
31900 study/epoch visits found min87, mean210.3747, max577; 39 exceeded the
initial proposed512 cap. No visit reaches640. The unadmitted draft was revised
before review/admission, with its old registration backed up.

Keep raw images on CPU. Transfer/augment32 windows at a time. Save stochastic
encoder RNG, keep embeddings after encoder plus spatial pooling, backpropagate
one full head loss, replay each encoder microbatch, then clip/step/schedule/EMA
once. Preserve the post-head RNG. BatchNorm or forward-mutated buffers fail.
The deployed Net and OFLEncoder ASTs are unchanged.

The registered comparison is macro AUC >= sealed seed0 on at least3/4 rulers
(`jev_hybrid`, `aquila_v1`, `dread`, `consensus`), each with all12 targets defined
on the exact977 own holdout. Invalid rows, hashes or receipts fail. This is an
EXPOSED_EXPLORATORY screen, clean_oof=false; different hosts and window exposure
confound attribution. It is not a matched-exposure experiment or submission
qualification. R7 complement comparison is NOT_RUN: this is the own-holdout
fullbag-versus-seed0 objective screen.

## Offline Linux overlay

Artifacts: `.local/scratch/astra-b2/k1-fullbag-v1/`.
`k1-fullbag-v1.tar`, `.tar.sha256`, `FULLBAG-SHA256SUMS`,
`FULLBAG-PREREQUISITES-SHA256SUMS`, `FULLBAG-COPY-LIST.txt`,
`FULLBAG-INSTALL.json`, `PACKAGE-RESULT.json`, `BUILD.diff`.
Exact destination copy list (hashes in FULLBAG-SHA256SUMS and
FULLBAG-INSTALL.json):

```text
tools/l1_rco_k1_fullbag_v1.py
tools/l1_rco_k1_fullbag_replay_v1.py
tools/l1_rco_k1_fullbag_check_v1.py
tools/l1_k1fullbag_trainer_v1.py
tools/test_l1_rco_k1_fullbag_v1.py
tools/l1_rco_k1_fullbag_v1.md
tools/k1_fold_dataset_v1.py
.local/scratch/astra-b2/k1-fullbag-v1/install_overlay.py
.local/scratch/l1-rco/k1v3-grouping/compact_train_l1.py
.local/scratch/l1-rco/k1v3-grouping/registration.json
.local/scratch/l1-rco/fullcache/registration.json
.local/scratch/astra-b4/H-B-146/launch-v115/registration.json
artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/folds.csv
```

Archive-only installation metadata:

```text
FULLBAG-COPY-LIST.txt
FULLBAG-INSTALL.json
FULLBAG-PREREQUISITES-SHA256SUMS
FULLBAG-SHA256SUMS
```

Source plus sealed roster/split metadata only: no images, target labels,
weights, predictions, review PASS, run receipts, job state or waiters.
B transfers it; builder uses no network. The five parent artifacts above are
copied from local sealed bytes: the trainer was missing from the fold package;
the other four are already in device-plan-v4 and are skipped when identical.
Any different existing parent fails before writes; none is regenerated.

Use the same project root and interpreter as f3. Existing initialization,
targets, cloud_code tree, cache and host qualification evidence remain required.
Prerequisites cover registration/fullcache-bound pins, imported runtime helpers,
and serving source; they exclude files supplied by this overlay.
`k1n_config_v1.py` is a **packager import**, not a training dependency: retain
device-plan-v4 v1.6.0 SHA256
`9a35f5b6cfd82b53ecdabdaf03ef41c5c3539c56c36a4708c162afc09aa2051b`.
Its transitive `c85r7_rco_overlay_v1.py` import is also pinned.
Do not copy local v1.7.1 or overwrite the installed foldgate. `foldgate` and
`exp156_k_sweep` are used only by optional `evaluate`, outside this installation
gate. Post-fit evaluation must separately qualify those dependencies.
Installed-source provenance for every prerequisite is in PACKAGE-RESULT.json.
SHA256SUMS and `.sha256` files use LF.

```bash
set -euo pipefail
task=.local/scratch/astra-b2/k1-fullbag-v1
mkdir -p "$task/stage"
(cd "$task" && sha256sum -c k1-fullbag-v1.tar.sha256)
tar -tf "$task/k1-fullbag-v1.tar"
tar -xf "$task/k1-fullbag-v1.tar" -C "$task/stage"
(cd "$task/stage" && sha256sum -c FULLBAG-SHA256SUMS)
sha256sum -c "$task/stage/FULLBAG-PREREQUISITES-SHA256SUMS"
python -B "$task/stage/$task/install_overlay.py" --root "$PWD" --stage "$task/stage"
# Only after PREFLIGHT_PASS; same preflight repeats before any write.
python -B "$task/stage/$task/install_overlay.py" --root "$PWD" --stage "$task/stage" --apply
```

Do not repeat installation over an attempted job or change an existing draft
registration on Linux. Any changed hash is HOLD pending versioned resolution.
The only existing executable changed by this overlay is the packager, from the
exact installed inventory hash to1.2.0. Its Linux backup is
`.local/backups/astra-b2/k1-fullbag-v1/k1_fold_dataset_v1.py.linux-before`.
An identical file is never rewritten. Missing parents are added before any
fullbag preparation. Fullbag attempt markers fail installation closed.
The live f2/all/f3 modules remain byte-identical.

## CPU preparation and local tiny numerical check

Run Linux preparation with the existing complete cache. If cache I/O competes
with training, do the real CPU smoke after f3 termination. None of these
commands launch CUDA.

```bash
cache=.local/scratch/astra-b2/h224-folds-v1/cache
python -B tools/l1_rco_k1_fullbag_v1.py register --mode fullbag --cache "$cache"
python -B tools/l1_rco_k1_fullbag_v1.py prepare --mode fullbag --cache "$cache"
python -B tools/l1_rco_k1_fullbag_v1.py validate --mode fullbag --cache "$cache"
python -B tools/l1_rco_k1_fullbag_check_v1.py --device cpu
python -B tools/l1_rco_k1_fullbag_v1.py cpu-smoke --mode fullbag --cache "$cache"
python -B tools/l1_rco_k1_fullbag_v1.py admit --mode fullbag --dry-run
```

Expect registration3190/977, CPU numerical PASS, CPU smoke PASS with CUDA
uninitialized. Before qualification and independent review, admit is BLOCKED.
Check JSON status; a dry-run's zero exit code is not admission.

**B only, local4070, after verified f1 terminal and no other GPU job:** this
explicit small arithmetic check uses toy bags plus crop64 ConvNeXt, not a fit.
Do not run it alongside f1 or start fullshape locally.

```powershell
& C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe -B tools/l1_rco_k1_fullbag_check_v1.py --device cuda
```

Require PASS; copy only `numerical-cuda.json` to the same task directory on
Vast with identical code pins. Do not copy Windows registration/preparation/
CPU-smoke or synthesize a Linux review. CPU/GPU numerical checks compare every
gradient, loss, AdamW state, scheduler, EMA and next RNG against ordinary
autograd on the same microbatch graph, variable lengths/masks, dropout,
drop-path and FP32/BF16. Tolerances: FP32 atol2e-5/rtol2e-4;
BF16 atol2e-3/rtol.03. These do not certify fullbag memory.

## Independent qualification, then bound f3 SUCCESS

No waiter is installed. Preserve local f1 and the entire existing Vast chain.
Reuse the existing full-cache META_EPS receipt; never rewrite it.
Runtime stack/TF32/SDPA parity and the existing host/cache proof must remain
valid; a changed host or stale proof is HOLD, not a reason to fabricate PASS.

```bash
python -B tools/l1_rco_k1_fullbag_check_v1.py --qualification-binding > "$task/qualification-binding.json"
```

Require PASS. An actual independent reviewer writes
`$task/REVIEW-QUALIFY.json`: `verdict`, `builder`, distinct `reviewer`,
project-relative `report`, `report_sha256`, and `qualification` equal to the
exact object from that binding. This review authorizes only the two-step
fullshape diagnostic below, not training.

**B only on Vast GPU0 after f3 SUCCESS:** read only canonical job metadata.
Qualification itself checks exact f3 command/run/owner and terminal receipts,
no live child, canonical GPU ownership, independent review and no prior
qualification attempt. Do not launch on a predicted finish time.

```bash
py="$(python -B -c 'import sys; print(sys.executable)')"
f3_run_id="$("$py" -B -c 'from tools import l1_rco_k1_fullbag_v1 as f; print(f.k.read(f.CANONICAL_OWNER/"artifacts/logs/H224_K1_V3G_F3/job.json")["run_id"])')"
export KNEE_WORKSPACE_ROOT="$PWD/.local/scratch/astra-b2/h224-folds-v1/gpu0-owner"
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
"$py" tools/gpu_job.py --tag DA052_K1_FULLBAG_QUALIFY -- "$py" "$PWD/tools/l1_rco_k1_fullbag_check_v1.py" --device cuda --full-shape --f3-run-id "$f3_run_id"
"$py" -B -c 'from tools import l1_rco_k1_fullbag_v1 as f; from tools import job_runtime as r; print(r.job_status(f.CANONICAL_OWNER/"artifacts/logs","DA052_K1_FULLBAG_QUALIFY"))'
```

Keep the exact absolute argv above; adding `-B` to the child changes the bound
command. Require bound SUCCESS plus `fullshape-cuda.json` PASS. The diagnostic
uses batch2 x cap640 x crop352, streamed32, two AdamW/EMA updates; real encoder/
head and synthetic images. Peak reserved <=12GiB and baseline used + peak +
2GiB <= total. A15min no-progress watchdog and durable once marker apply.
Failure/interruption means HOLD; no retry or deleting receipts.

```bash
python -B tools/l1_rco_k1_fullbag_v1.py review-binding --mode fullbag > "$task/fullbag-bindings.json"
```

Require PASS. The independent reviewer then writes a separate
`$task/REVIEW-META-EPS.json` with `verdict`, `builder`, distinct `reviewer`,
`report`, `report_sha256`, exact `bindings`, exact `meta_eps_bindings`,
and exact `fullbag_memory_sha256` from this output. Review actual Linux CPU
evidence, CUDA numerical evidence, bound qualification SUCCESS, measured
memory/step time, unchanged source/runtime and predecessor identity.

```bash
python -B tools/l1_rco_k1_fullbag_v1.py admit --mode fullbag --dry-run
python -B tools/l1_rco_k1_fullbag_v1.py launch-after-f3 --mode fullbag --f3-run-id "$f3_run_id" --dry-run
# B ONLY, after CPU_ADMISSIBLE and PLAN_ONLY and actual independent PASS:
python -B tools/l1_rco_k1_fullbag_v1.py launch-after-f3 --mode fullbag --f3-run-id "$f3_run_id"
```

No polling, time-trigger or automatic retry. The sequence locks the new
`after-f3-once.json`, revalidates predecessor and review under lock, records
STARTED before dispatch, uses the same canonical GPU0 owner, and binds its
fresh job receipt through SUCCESS. STOPPED is terminal for this attempt.
Direct `run` cannot bypass the sequence; `infer` requires its owned production
parent. The inherited production heartbeat/stall/throughput guards apply.

## Post-fit exports and objective screen

These commands require completed production artifacts (`cpu_test=false`).
The builder cannot produce them without the prohibited fit. The production
run supervises train/infer/export; no separate GPU inference is needed here.

```bash
python -B tools/l1_rco_k1_fullbag_v1.py export --mode fullbag
python -B -m tools.k1_fold_dataset_v1 verify-export --export "$task/EXPORT.json"
python -B -m tools.k1_fold_dataset_v1 package --export "$task/EXPORT.json" --output "$task/dataset-k1-v3gfb-v1" --dataset-id taeyangg4/knee-k1-v3gfb-v1-assets
python -B -m tools.k1_fold_dataset_v1 verify-dataset --dataset "$task/dataset-k1-v3gfb-v1"
```

Optional objective evaluation is separate from fit/qualification/infer/export/
package. Before `python -B tools/l1_rco_k1_fullbag_v1.py evaluate --mode fullbag`,
independently check the installed foldgate/rulers, ruler inputs and host binding.
This repair does not install the local host-bound foldgate fix or claim the old
Vast foldgate qualifies for that later action.

Output is the standard fold export layout, fold0 checkpoint and frozen serving
source, model ID `k1_v3g_fullbag`. No upload command is part of this card.
The packager refuses an existing destination; verify it on rerun, never delete
it. Own-holdout CSV/result are in `$task/fit/holdout.csv` and `result.json`,
and hash-bound by `$task/EXPORT.json`. `comparison.json` contains aggregate
four-ruler metrics and the registered decision, never label rows. Independent
result review and T4 serving parity remain required before downstream use.

## Time estimate and builder evidence

Metadata census: median199, p90=297, p95=349, p99=409 windows.
Relative candidate exposure3.2871x; illustrative paired/padded32 exposure3.4078x.
Using B's measured capped64 4090 time0.55s and an **unmeasured** replay overhead
factor1.25..1.5 gives2.34..2.81s/update,10.38..12.46h training-only for15950
updates. Validation/I/O/checkpoint time is additional; this is not a completion
promise. Worst-case cap640 qualification is deliberately larger than normal
bags. Replace estimates with measured qualification/production telemetry.

See `.local/scratch/astra-b2/k1-fullbag-v1/BUILD-RESULTS.md` for actual CPU
results and limits. Linux execution, CUDA checks, independent review, trained
weights, production977 predictions and AUCs are PENDING. None is implied by
CPU or synthetic receipt tests.
