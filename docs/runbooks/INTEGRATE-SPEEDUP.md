# H-B-221: K1 x3 runtime v6 integration

Builder: astra-b1. One candidate and its diagnostic companion. CPU build only;
independent review, local GPU benchmarking and Kaggle qualification remain pending.

## Registered change

`tools/k1sr_runtime_v6.py` transfers the cropped study tensor, metadata and mask
once, then reuses those device tensors for all three frozen K1 models. The release
constant `ENCODER_CHUNK = 32` is not environment-selectable. Chunk 64 is exposed
only to the benchmark; accepting it requires a separately registered version and
stage after FP16 parity and memory review. No chunk64 stage is included here.

The per-target rank mean and outer 80/20 rank blend, renderer, checkpoint identities,
FP16 settings, parent-first order and process hooks are unchanged. The 1500 s
per-model load budget, combined 120 s transfer/classification budget and 8 h guard
remain. Existing receipt fields and exact-parent-byte fail-safe remain; study and
aggregate receipts add transfer timing and the registered chunk. Per-model classify
timing now excludes the separately reported transfer. `classifier_seconds` includes
transfer, so its timeout accounting stays comparable with v5.

## Immutable stages

| Purpose | Stage under `artifacts/experiments/EXP169_RELEASE_C85R7/` | Kernel slug |
|---|---|---|
| Release | `v1.1.64-r6w958k1x3sr20` | `taeyangg4/knee-lb-r6w958k1x3sr20-v3` |
| N32 diagnostic | `v1.1.64-r6w958k1x3sr20-n32` | `taeyangg4/knee-lb-r6w958k1x3sr20-v3-n32` |

Both stages include `registration.json`, `v5-payload-scope.json` and the full
`v5-full-payload-notebook.diff`. The scope check compares every embedded file,
the complete notebook structure and metadata against the existing v1.1.62 stages.
Only the runtime replacement, its import reference, chained hashes, version/slug
and explicit v6 configuration are allowed. Unchanged member hashes are recorded.
Runtime v1-v5, shared renderer and predecessor stages remain unchanged.

N32 uses the existing label-free roster preparer. Both embedded parent collection
and the K1 overlay suppress `submission.csv`; receipts carry `never_submit=true`,
`qualified_for_submission=false` and null submission hashes. Never submit N32.

## CPU launch card (executed by builder)

From the project root in PowerShell:

```powershell
$py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
$env:CUDA_VISIBLE_DEVICES = '-1'
& $py -m tools.k1sr_benchmark_v6 validate
& $py -m tools.k1sr_cpu_smoke_v6
& $py -m tools.c85r7_r6w958k1x3sr_v1164 prepare k1x3sr20
& $py -m tools.c85r7_r6w958k1x3sr_v1164 validate k1x3sr20
& $py -m tools.c85r7_r6w958k1x3sr_v1164 prepare k1x3sr20 --n32
& $py -m tools.c85r7_r6w958k1x3sr_v1164 validate k1x3sr20 --n32
```

`prepare` has already created the stages and intentionally rejects an existing
destination. Reuse `validate`; do not overwrite a stage. Registration is emitted
by prepare; there is no separate register/admit dry-run in this builder. Benchmark
`validate` is CPU-only identity/roster preflight, not GPU resource admission.

Evidence is under `.local/scratch/astra-b1/h-b-221/`: `benchmark-validation.json`,
`stage-*-prepare.json`, `stage-*-validate.json`, `tests-full-pinned.txt`,
`cpu-smoke.log` and `cpu-smoke/receipt.json`. Detailed results: `BUILD-REPORT.md`
in that directory. CPU replay establishes neither CUDA FP16 equivalence nor speed.

Completed evidence: 248-test full regression PASS, then 18-test final speedup
suite PASS after benchmark guard additions. Real-DICOM CPU smoke PASS: one study,
127 windows, all three models; chunk32 and chunk64 each matched v5 exactly
(maximum absolute difference 0 for all models). Fallback and diagnostic publication
checks passed. Original-file inventory: 19/19 hashes unchanged.

## B-only GPU benchmark, after the current stride screen

Obtain an independent review of the exact files and CPU evidence. The reviewer
must create `.local/scratch/astra-b1/h-b-221/GPU-REVIEW.json` with `verdict: "PASS"`,
`builder: "astra-b1"`, their distinct nonempty `reviewer` identity and an `identity`
object exactly matching `benchmark-validation.json`. No approval file is supplied
by this build. Both CLI and direct benchmark entry reject absent review; stale
source/roster identities and self-review are rejected.

B must confirm the existing stride job is terminal and claim/admit the single GPU
through the existing queue/resource procedure before this one launch. Do not add
a competing waiter or replace an existing queue reservation.

```powershell
$py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
& $py tools/gpu_job.py --tag h-b-221-k1x3-v6-benchmark -- $py -m tools.k1sr_benchmark_v6 run --review .local/scratch/astra-b1/h-b-221/GPU-REVIEW.json
```

This command was NOT executed by the builder. The benchmark uses the existing
32-study harness roster and pinned DICOM bytes/checkpoints; no labels/downloads.
One common CPU rendering feeds v5, v6 chunk32 and v6 chunk64. All models are
resident, first-study warmups are excluded, arm order rotates, CUDA timings are
synchronized and allocator peaks reset per arm. Rendering, parent inference and
model loading are outside the timed comparison; loading is reported separately.
The latest complete study records survive watchdog termination in
`GPU-BENCHMARK.json`; watchdog failure has its own JSON file.

The report includes each study's total/transfer/model times, peak allocated and
reserved memory, all three model outputs, per-target correlations and rank-mean
comparisons. Gates for each candidate arm: max absolute difference <= 0.005 and
all 12 Spearman correlations >= 0.999 versus v5, for each model and the rank mean.
Undefined correlations fail. Peak reserved must be <= 12 GiB, and measured
non-model device usage + peak reserved + 2 GiB must fit the card. Overall PASS
requires both experimental arms; assess each arm separately if chunk64 fails.
The engineering gate is numeric/memory qualification, not a guaranteed speedup.
Require independent result review before adoption. No automatic retry/admission.

## Expected saving and remaining qualification

The change eliminates two of three repeated input transfers. It does not remove
two thirds of classifier computation. Actual seconds/hours saved are unmeasured.
For measured mean classifier-path times `t5` and `t6`, the report estimates
`(t5 - t6) * 1300 / 3600` hours saved; a negative value means a regression.
For scale only, 1 second saved per study is about 21.7 minutes over 1300 studies.
Chunk64 may improve encoder throughput but is unqualified and may increase memory.

The supplied historical estimate is INF3 6.7-6.9 h plus K1 x3 about 1.5 h, or
roughly 8.2-8.4 h total. No measured saving or <=9 h hidden-runtime claim follows
from this CPU build or from a 4070 K1-only benchmark. Full-parent T4 N32 timing,
ordinary visible output, fallback/codecs and independent minimum release review
are still required under B's existing qualification process. Do not rewrite the
formula or choose chunk64 from timing alone.

## B-only push commands (not executed)

After independent review and successful local engineering evidence, refresh live
settings/quota/duplicates and push serially. Run the diagnostic first, inspect
full-pipeline timing/output, and qualify the release with its ordinary visible run.

```powershell
kaggle kernels push -p artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.64-r6w958k1x3sr20-n32/kernel
# After the diagnostic completes and its evidence is accepted:
kaggle kernels push -p artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.64-r6w958k1x3sr20/kernel
```

These are push commands only. Any later qualified release submission must use
`tools/kaggle_submit_once.py` with B's exact live version/ledger checks. N32 must
never enter that path. Builder performed no GPU, Kaggle, network or sub-agent work.
