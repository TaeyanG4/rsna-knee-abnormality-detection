# S6+ v1.0.1 - H-B-108 LAUNCH card

New builder; public sources and scored stages are unchanged. B owns all launches.
Registration: `artifacts/experiments/S6PLUS_V1/registration.json`.
Config: `configs/s6plus-v1.json`. Builder: `tools/s6plus.py`.
Contract: `a05a99e47fa815ec73a4aa858c46fbf71a8528f840a489dff1d1e1e33eb9e4d3`.

Changelog 1.0.1: restore public spatial-attention and extent heads (present in
all four checkpoints despite missing manifest flags), nine metadata columns,
pretrained spatial-head normalization, and checkpointed spatial pooling.
The development v1.0.0 files are backed up under
`.local/backups/s6plus-v100-pre-spatial-fix/`. Public source files are unchanged.

## Frozen comparison

EXP099 fold 0: 3,455 training / 888 held-out studies. Patient components and
report-group hashes are disjoint. The two known corrupt studies are excluded.
S-B uses Dread; S-L uses EXP143 v2 L2 (0.75 Dread + 0.25 aquila); S-LD adds
public S6 categorical KL and auxiliary binary BCE, temperature 2, weight 0.25.
Four public members are averaged as probabilities; targets exist only for the
training roster. **S-LD is production diagnostic: the public teacher saw fold 0.**
The side classifier's ancestry is not established fold-clean, so even S-B/L
require that caveat. No OAI-panel eligibility or superiority over public S6 is claimed.

All arms start from the identical ImageNet-22k/1k ConvNeXt-S initialization, use
the public spatial/extent heads, and share 384/352 anatomy/side-aware v4 crops,
three slices at 3.5 mm, 176 training
windows, batch 2, 16 epochs, AdamW encoder/head LR 1e-4/1e-3, weight decay .05,
5% warmup/cosine, clip 5, drop-last, final epoch only. This is 27,632 updates.
EMA and sampling are absent from the public manifest: source defaults 0/uniform
are frozen prospectively, with crop-only augmentation and no slice jitter.
Original report categories/certainty/masks are unavailable: B/L use soft macro-12
BCE, not the public original supervision. The public pool priors are replaced
by shared Dread training-only marginals; category priors are neutral and the
state-evidence gate is frozen at zero. LD supervises actual category heads.

Encoder and spatial-pooling chunks of eight use activation checkpointing; every real window still
enters one full-study MIL head. No batchnorm exists in this encoder. Checkpoint
every 50 updates/epoch/bounded stop, with latest + previous + transient file;
optimizer, scheduler, crop/data RNG and next batch resume exactly. All paired
arms must use one identical GPU/software/precision stack. Do not split the arms
between 4070 and 3090. Inference parts and teacher targets resume per study.

## CPU setup and preparation

Run from the repository root in PowerShell. No package installation is needed.

```powershell
$py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '2'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:S6PLUS_PIN_DLL = 'C:\Windows\System32\DriverStore\FileRepository\nv_dispui.inf_amd64_49a28c45e7d7dc62\nvdxgdmal64.dll'
& $py tools/s6plus.py config
& $py tools/s6plus.py register
& $py tools/s6plus.py validate
& $py tools/s6plus.py cache --workers 2 --threads 2 --dry-run
& $py tools/s6plus.py cache --workers 2 --threads 2
```

The builder executed `config`, `register`, `validate`, the full-cache dry-run,
and the four-study real cache smoke below. The full cache build is a launch
step for B, not a completed builder artifact. Re-running registration must
match exactly; never overwrite a frozen registration to accept a code change.

The opt-in DLL pin is a process-local workaround for the observed Windows
`nvdxgdmal64.dll_unloaded` CPU-backward crash. It changes no driver and initializes
no CUDA; its hash is recorded in execution receipts. Omit it on Linux. A different
Windows driver requires its observed installed path. Keep the choice fixed across
paired runs. Cache workers use four threads total; native anatomy/side run on CPU.
Raw DICOMs are extracted one study at a time with ZIP CRC checks. Never open raw
`train.csv`: a UID-only substitute is used. Per-study compressed uint8 shards
contain all S6 views; full cache requires both training and held-out rosters.

For dry-runs before launch (no accelerator work):

```powershell
& $py tools/s6plus.py teacher --device cuda --dry-run
foreach ($arm in @('S-B','S-L','S-LD')) {
  & $py tools/s6plus.py train --arm $arm --out "artifacts/experiments/S6PLUS_V1/runs/$arm" --device cuda --dry-run
  & $py tools/s6plus.py collect --run "artifacts/experiments/S6PLUS_V1/runs/$arm" --out "artifacts/experiments/S6PLUS_V1/runs/$arm/oof.csv" --device cuda --dry-run
}
& $py tools/s6plus.py prepare --out .local/s6plus-v1-portable.zip --dry-run
& $py tools/s6plus.py admit --run artifacts/experiments/S6PLUS_V1/runs --dry-run
& $py tools/s6plus.py screen --run artifacts/experiments/S6PLUS_V1/runs --out artifacts/experiments/S6PLUS_V1/screen --dry-run
```

An admission dry-run reporting missing OOF/checkpoints is expected before fits;
it is not authorization or a successful result. Full admission requires exactly
888 valid probabilities per arm and all 27,632 updates, matching hashes and stack.

## B-only CUDA smoke and fits

These are **instructions only**, not executed by the builder. Check/claim the GPU
lane first; no inference or diagnostics alongside training. Remove the CPU mask:

```powershell
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
& $py tools/gpu_job.py --tag s6plus-cuda-smoke --wait -- $py tools/s6plus.py train --arm S-B --out .local/s6plus-cuda-smoke/S-B --device cuda --smoke --cuda-smoke --workers 0
if ($LASTEXITCODE -ne 0) { throw 'CUDA smoke failed' }
```

This uses real 352-pixel inputs, batch 2 and up to 176 windows per study. Read
`updates.jsonl` for seconds/update, actual window count and peak CUDA memory.
CUDA smoke is explicitly not collectable/admissible. Measure the first 50 full
updates and use their elapsed time to replace the planning GPU estimate.

```powershell
& $py tools/gpu_job.py --tag s6plus-teacher --wait -- $py tools/s6plus.py teacher --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Teacher generation failed' }
foreach ($arm in @('S-B','S-L','S-LD')) {
  & $py tools/gpu_job.py --tag "s6plus-$arm-first50" --wait -- $py tools/s6plus.py train --arm $arm --out "artifacts/experiments/S6PLUS_V1/runs/$arm" --device cuda --workers 2 --stop-after-updates 50
  if ($LASTEXITCODE -ne 0) { throw "Fit failed: $arm" }
  & $py tools/gpu_job.py --tag "s6plus-$arm-rest" --wait -- $py tools/s6plus.py train --arm $arm --out "artifacts/experiments/S6PLUS_V1/runs/$arm" --device cuda --workers 2 --resume
  if ($LASTEXITCODE -ne 0) { throw "Fit failed: $arm" }
  & $py tools/gpu_job.py --tag "s6plus-$arm-oof" --wait -- $py tools/s6plus.py collect --run "artifacts/experiments/S6PLUS_V1/runs/$arm" --out "artifacts/experiments/S6PLUS_V1/runs/$arm/oof.csv" --device cuda
  if ($LASTEXITCODE -ne 0) { throw "Collect failed: $arm" }
}
$env:CUDA_VISIBLE_DEVICES = '-1'
& $py tools/s6plus.py admit --run artifacts/experiments/S6PLUS_V1/runs
if ($LASTEXITCODE -ne 0) { throw 'Full admission failed' }
& $py tools/s6plus.py screen --run artifacts/experiments/S6PLUS_V1/runs --out artifacts/experiments/S6PLUS_V1/screen
```

On interruption, resume the same arm with `--resume`; repeat teacher/collect to
reuse verified parts. Never change the registration to accept changed artifacts.
The cache, CPU smoke teacher directory, and full teacher directory have different
purposes: smoke targets cannot be used by a full fit.

## Readout and optional transfer

`screen` invokes `tools/multi_lens_readout.py` for S-L vs S-B and S-LD vs S-L:
paired component bootstrap, every target/ruler and weak-four/macro-12. One fold
and one seed cannot establish replication. Missing frozen, fold-clean partner
OOFs make the blend lens unavailable; supply B's registered partners to the
emitted specs before running that optional lens. Gold58 is descriptive only.
Smoke readout is plumbing validation and has no efficacy interpretation.

`prepare` creates a ZIP_STORED portable cache/weights/source/labels/split package
without raw DICOMs, report text or credentials. It does not upload anything.
Only run a full package after the cache finishes:

```powershell
& $py tools/s6plus.py placement --out artifacts/experiments/S6PLUS_V1/placement.json
& $py tools/s6plus.py prepare --out .local/s6plus-v1-portable.zip
```

Unpack into a dedicated remote workspace and use its Python with existing
dependencies. Run `tools/s6plus.py validate`, then pass `--cache cache` for teacher,
train and collect. The package supports training/inference from the cache; it
does not contain native DICOM cache-building dependencies or the local multi-lens
reference tables. Collect remotely and run multi-lens locally. No fresh register
is needed remotely. Keep all three arms on that same remote stack if chosen.

## Placement decision

**Use the local 4070 for all three fits, sequentially on one software stack.**
Four actual studies (660 DICOMs) produced 207.72 MB compressed cache. Study-level
CPU times were 14.32-22.79 s with two threads per worker. File-count extrapolation:

| Scope | Compressed cache | Serial CPU | Ideal two-worker CPU | Upload at 0.8-1.1 MB/s |
| --- | ---: | ---: | ---: | ---: |
| 3,455 training studies | 202.57 GB | 18.78 h | 9.39 h | 51.15-70.34 h |
| Training + 888 held-out | 253.99 GB | 23.55 h | 11.78 h | 64.14-88.19 h |

Allow 0.5-2x extrapolation uncertainty plus parallel/I/O overhead. Reserve roughly
260 GB additional cache space; packaging a full portable ZIP needs another
approximately 255 GB. The cache is compressed uint8 but still too large for a
quick Vast transfer at the observed uplink. Its 2.7-3.7 day upload dominates the
choice unless local queue delay exceeds the remote compute advantage or the
cache is already remote. Reconcile this with H-B-109/R+ local scheduling.

GPU throughput is **unmeasured**. At 27,632 updates per fit, assumptions of
2/4/8/12 s per update yield 15.35/30.70/61.40/92.11 GPU-hours per fit. Plan
30.7-92.1 h per fit (92-276 h for the three), excluding four-member teacher
inference and 888-study OOF. This is a planning range, not a 4070 or 3090
benchmark. Replace it using the full-window CUDA smoke and first 50 updates;
measure teacher/OOF throughput separately. CPU smoke timing does not predict
GPU speed. This estimate supersedes the consultation's optimistic schedule.

## Repeatable CPU smoke and tests

Use the CPU environment above. These commands were executed successfully for
v1.0.1; repeat in a fresh smoke output directory, or explicitly use `--resume`
for an existing training run. Smoke never qualifies as an 888-study result.

```powershell
& $py -c "import tools.s6plus; import pytest; raise SystemExit(pytest.main(['tests/test_s6plus.py','tests/test_multi_lens_readout.py','tests/test_exp189_s6_port.py','--basetemp=.local/pytest-logs/s6plus-tmp','-q']))"
& $py tools/s6plus.py cache --smoke --workers 2 --threads 2
& $py tools/s6plus.py teacher --smoke --device cpu --teacher-dir storage/cache/s6plus-teacher-smoke-v1
foreach ($arm in @('S-B','S-L','S-LD')) {
  & $py tools/s6plus.py train --arm $arm --out ".local/s6plus-smoke-v101/$arm" --smoke --device cpu --workers 0 --teacher-dir storage/cache/s6plus-teacher-smoke-v1
  if ($LASTEXITCODE -ne 0) { throw "CPU smoke failed: $arm" }
  & $py tools/s6plus.py collect --run ".local/s6plus-smoke-v101/$arm" --out ".local/s6plus-smoke-v101/$arm/oof.csv" --smoke --device cpu
  if ($LASTEXITCODE -ne 0) { throw "CPU collection failed: $arm" }
}
& $py tools/s6plus.py admit --run .local/s6plus-smoke-v101 --smoke
& $py tools/s6plus.py screen --run .local/s6plus-smoke-v101 --out .local/s6plus-smoke-v101/screen --smoke
& $py tools/s6plus.py prepare --smoke --out .local/s6plus-v101-portable-smoke.zip
```

Portable extraction verification checked all 30 member hashes and reran the
actual CPU teacher -> S-LD update -> collect in the extracted workspace. Its
OOF and checkpoint SHA256 exactly matched the local run. The extracted package
contains label tables; verification logs report only counts and hashes.

## Verification record

See `.local/astra-improve/builds/s6plus-build-out.md`, its log links, and
`artifacts/experiments/S6PLUS_V1/placement.json`. Full 888-study OOF is produced
only after B executes the fits. No GPU training was launched by this builder.
All CPU launch commands were exercised as actual smokes or full-scope dry-runs;
full admission correctly remains false while full fits/OOFs are absent.
Hashes: `artifacts/experiments/S6PLUS_V1/SHA256SUMS.txt`.
