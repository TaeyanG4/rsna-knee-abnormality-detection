# K1 N-model runtime v7.1 integration

Changelog v7.1.0 / generator v1.1.0: registered K1 eval_stride 2/3,
raw-row adapter v2, stride in registration/render contract and receipts.

Builder: astra-b1. Config-only generation for 1..6 compatible K1 checkpoints.
CPU preparation is complete; independent review and GPU/hidden qualification are pending.

## Runtime and formula

`tools/k1sr_runtime_v7.py` preserves the v6 R6 renderer and FP16 forward.
`k1_eval_stride` defaults to 3 and accepts only integer 2 or 3. Adapter v2 uses
the identical frozen K1 `window_rows` function on captured raw px/meta/blocks;
it does not subsample the existing R6 windows. The parent stays at stride 1.
Side-corrected studies rerender with the original K1 side contract at the selected
stride, restoring the parent's fields and forced-side map afterward.
Each study's cropped input, metadata and mask are transferred once and reused by
every registered model. Encoder chunk is fixed at 32; 64 is refused because its
FP16 gate has not been measured. Model loads have separate 1500 s budgets; transfer
plus all classifiers share a 120 s study budget. The K1 wall guard remains 8 h.
These guards retain v6 behavior: native CUDA hangs cannot be forcibly preempted by
Python; late returns are rejected.

For full-roster, per-target average-tie percentile ranks `r`, define
`K = mean_i(r(model_i_probability))`.

- `drop_o=false`: `r(.80*r(INF3) + .20*r(K))`, identical x3 math to v1.1.64.
- `drop_o=true`: `r((1-alpha)*r(P) + alpha*r(K))`; alpha defaults to .25.
  `P` is the original INF3 P branch including R224, original survivor substitution
  and the R224 >3% failure abort. The CSV float32 boundary is retained.

The drop edit removes O asset preflight, the A172 child launch and O collection;
EXP143v2, EXP155 and Q_A5q inference is never invoked. Public R/S/R224 inference
sources are unchanged. The two O-only dataset mounts are removed. Any K1 model,
capture, timeout, journal or identity failure discards the entire K1 ensemble and
publishes the exact selected-parent bytes with `DEGRADED_PARENT_ONLY`. With drop_o,
that parent is P; without drop_o, it is INF3. Existing parent hard failures remain
hard failures.

`k1sr-receipt.json` retains parent/fallback/phase fields and records registered
model identities, config SHA, `k1_eval_stride`, transfer time, per-model load/classify time and
per-study outcomes. Success is `COMPLETE_SHARED_K1N`.

## Configs and immutable stages

| Config | Purpose | Stage / slug |
|---|---|---|
| `configs/k1n/current-x3.json` | Real v2, seed0, seed1; INF3 80/20 | `v1.1.65-r6w958k1n3sr20` / `taeyangg4/knee-lb-r6w958k1n3sr20-v1` |
| `configs/k1n/six-model-drop-o.placeholder.json` | Above plus fold1/2/3; P/K1 75/25 | Reserved example `v1.1.66-r6w958k1n6p25`; **not prepared** |
| `configs/k1n/x3-stride2-drop-o.json` | Real x3, stride 2, drop O, P/K1 75/25 | `v1.1.67-r6w958k1n3s2p25` / `taeyangg4/knee-lb-r6w958k1n3s2p25-v1` |

Stages live under `artifacts/experiments/EXP169_RELEASE_C85R7/`. `prepare` creates
the release and its `-n32` stage/slug together, including config, registration,
full payload/notebook diff and hashes of every untouched embedded file. Existing
destinations are refused. `validate` reconstructs both artifacts and checks the
complete inventory and every byte. Original runtimes v1-v6 and predecessor stages
were not edited. The existing v1.1.65 release/N32 pair is byte-identical (14 files
checked). Runtime v7, config/generator, tests and this document were backed up
before this follow-up under `.local/backups/astra-b1/k1n-stride-v710/`.
The v1 adapter is unchanged; v2 is a new file. Historical v1.1.65 was generated
with v7.0.0/generator1.0.0: current generator reconstruction intentionally rejects
that old payload as drift. Preserve its embedded runtime and historical evidence;
use new versions for all v7.1 configurations.

N32 is enforced never-submit: it produces `diagnostic-k1n.csv`, never
`submission.csv`, carries `never_submit=true`, and has a null submission hash.
Never submit this timing companion. Neither stage is marked qualified by preparation.

For tomorrow's folds, copy the placeholder config to a new registered config and
fill each real private dataset slug, checkpoint SHA256 and manifest SHA256. Verify
the trainer SHA256 as well; do not assume it matches the example. Each model entry
requires `id`, `dataset_slug`, `checkpoint`, `checkpoint_sha256`, `trainer_source`,
`trainer_sha256`, `manifest_sha256`. Model IDs, checkpoint hashes and mount basenames
must be unique. Pick unused release version and kernel slug before preparation.

The compact layout matches the current seed1 assets:

```text
asset-manifest.json
weights/E_prior_s0_f0.pt
source/compact_train.py
```

The manifest's `files` dictionary must contain exactly the checkpoint/source paths
and hashes. Optional `dataset-metadata.json` is allowed. Source and checkpoint
hashes are verified before import/deserialization. Trainer `Net` must match the
current ConvNeXt-small crop352, spatial=True, extent=False architecture, target
order and state keys/shapes/dtypes. The original v2 asset inventory has its exact
legacy identity exception. Arbitrary hashes are not evidence that assets exist:
prepare is offline; runtime authenticates the mounted files. Placeholder values
are refused before any destination is created.

## CPU launch card (executed)

From the project root in PowerShell:

```powershell
$py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
$env:CUDA_VISIBLE_DEVICES = '-1'
& $py -m tools.k1n_stage_v1 prepare --config configs/k1n/x3-stride2-drop-o.json
& $py -m tools.k1n_stage_v1 validate --config configs/k1n/x3-stride2-drop-o.json
& $py -m tools.k1n_stride_cpu_smoke_v1
```

The stride-2 x3 pair already exists; use `validate` now. Preparation itself registers the
pair. There is no separate register/admit command and no GPU admission was claimed.
After real fold identities are filled, the same prepare/validate commands accept
the new config path without code changes.

Expected negative check, executed (CLI exit 1, no stage; also exercised through the
generator API and tests):

```powershell
& $py -m tools.k1n_stage_v1 prepare --config configs/k1n/six-model-drop-o.placeholder.json
```

Full relevant suite, executed with the fixed lane temp directory:

```powershell
$env:S6PLUS_PIN_DLL = 'C:\Windows\System32\DriverStore\FileRepository\nv_dispui.inf_amd64_49a28c45e7d7dc62\nvdxgdmal64.dll'
& $py -c "from tools import s6plus; import pytest; raise SystemExit(pytest.main(['tests/test_k1_shared_render_v1.py','tests/test_k1sr_release_v1.py','tests/test_k1sr_release_v2.py','tests/test_k1sr_release_v3.py','tests/test_k1sr_release_v5.py','tests/test_k1sr_release_v6.py','tests/test_k1sr_load_timeout.py','tests/test_k1sr_speedup_v6.py','tests/test_k1n_v7.py','tests/test_k1n_stride_v710.py','--basetemp=.local/pytest-tmp/astra-b1','-q']))"
```

Historical v7.0 result: **331 passed**, one expected constant-input Spearman warning in the v6
rejection test. Tests cover N=1..6 rank math, x3 byte parity with v6, every model's
load/classification failure, one transfer, guards, malformed/partial outcomes,
drop-O orchestration and P survivor behavior, payload scope, immutable pair
validation and placeholder rejection.

Historical v7.0 real-data CPU smoke used one authenticated study from the existing 32-study
roster, 127 real R6 windows, the spawned loader and all three real checkpoints
in CPU FP32. It exercises release/N32, capture/model failure and guard paths.
The generated drop-parent release/N32 orchestration is replayed with real K1
predictions at R/S/R224 classifier boundaries. This is **not full parent inference**,
not CUDA/FP16 validation, and not execution of the unavailable three fold models.
O execution is forbidden in that replay and P output is checked exactly.

Evidence is in `.local/scratch/astra-b1/k1n-v7/`: `tests-full.txt`, `prepare.json`,
`validate.json`, `placeholder-result.json`, `cpu-smoke/receipt.json`,
`cpu-smoke-exit.txt`, `original-hashes-verified.json` and `BUILD-REPORT.md`.

Follow-up full suite: **378 passed, 1 expected warning, 41.25 s, exit 0**.
Prepare: `PREPARED_LOCAL_ONLY`; validate: `PASS_CPU_VALIDATE`, both exit 0.
Both release/N32 payload inventories pass. The additional full embedded diff
against v1.1.65 changes only config/runtime, drop-O parent and version metadata,
and adds adapter v2; all other embedded members are identical.

The v7.1 stride smoke runs the original K1 `CStudies(eval_stride=2)` and the
registered v7.1 `LoaderDataset` in separate CPU processes on the same authenticated
real study. Result: **177 windows; image and metadata hashes exactly equal**.
Original-side rerender and parent-state restoration also pass; CUDA was never
initialized. This follow-up does not rerun unchanged classifiers or claim FP16
parity. Synthetic tests execute the original CStudies method over edge/spacing
cases for both strides and cover invalid config types, receipt stride,
all-model failure/guards and N32 never-submit.

Follow-up evidence: `.local/scratch/astra-b1/k1n-stride-v710/`
(`tests-full.txt`, `prepare.json`, `validate.json`, `cpu-smoke/comparison.json`,
`frozen-stage-verified.json`, `followup-payload-scope.json`,
`v165-to-v167.diff`, `v165-to-v167-n32.diff`, `placeholder-result.json`,
`placeholder-cli.txt`, `BUILD-REPORT.md`).

## Qualification still needed

B must obtain independent review and execute the actual configured models on the
4070 after the current screen. Check FP16 parity, peak memory, full-parent output,
fallbacks and N32 timing for the exact stage before qualification. Six resident
models increase memory and classifier time; CPU tests cannot establish GPU fit.
No GPU, Kaggle or network command was executed by this build.

Using the supplied O share of approximately 17%, dropping O from a 6.7-6.9 h INF3
parent suggests roughly **1.1-1.2 h gross saving**. This is an estimate, not a
measurement of v7. Added fold classifiers offset it; one-transfer savings remain
unmeasured here. Do not infer a <9 h hidden runtime from this estimate.

D-B-247 is the supplied reason for the stride-2 example: x3 gain +.0007..+.0017
on the four 976 rulers, with approximately 1.44 times as many windows. This is
exposed exploratory evidence, pending independent confirmation. The larger K1
window count offsets part of the estimated O saving; no net hidden-time saving
is established for this stage. Measure its N32 timing before qualification.

SUB-AGENTS: none.
