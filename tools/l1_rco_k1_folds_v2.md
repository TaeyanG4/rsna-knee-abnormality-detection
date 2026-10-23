# FOLDS.md — H-B-224 K1 v3-G folds, v2.0.0

Changelog v2.0.0: one Vast 4090, sequential f2 then f3 on GPU0; bound SUCCESS required. Training contracts and v1 sources are unchanged. Supersedes the v1 launch-pair instructions.

Builder: astra-b2. B owns all GPU launches. Offline additive build; no original
run, frozen source, admission or receipt is changed. Independent review is pending.
This is one registered fold-family infrastructure task, not three tuned recipes.

## Contract and outputs

Parent: `.local/scratch/l1-rco/k1v3-grouping/registration.json`, contract
`a58f1476ea4ca3c8d64fe06c80bc6e24d78fdf2f347c4de68c1eba5d9c2ea83d`.
Holdout: `int(sha1(uid.encode()).hexdigest()[:8],16) % 4 == k`.
Same 4,167-study universe, labels, initialization, v3 optimizer grouping, seed0,
batch2, training64 windows, crop352, ten epochs, one-epoch warmup, final EMA,
collapse/watchdog policy and original inference. Fold-dependent RNG formulas
remain the original formulas; training-only prior bias is recomputed per fold.
Gold is excluded and known connected components may not cross the fold boundary.

| Fold | Host | Train | Holdout | Updates (10 epochs) | 4070 estimate |
|---|---|---:|---:|---:|---:|
| 1 | local Windows 4070 Ti SUPER | 3073 | 1094 | 15360 | 13.23 h |
| 2 | Linux 4090 GPU0 | 3100 | 1067 | 15500 | 13.35 h |
| 3 | Linux 4090 GPU0 (after f2) | 3138 | 1029 | 15690 | 13.51 h |

The common task root is `.local/scratch/astra-b2/h224-folds-v1` (called TASK below).
Per-fold outputs: `TASK/f{k}/fit/models/E_prior_s0_f{k}.pt`, `fit/holdout.csv`,
`fit/result.json`, `fit/runtime.json`; `f{k}/EXPORT.json` and `SHA256SUMS` bind
the model, predictions and provenance. No automatic retry of consumed admissions.

## Local CPU preparation (PowerShell, project root)

```powershell
$py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
$task = '.local/scratch/astra-b2/h224-folds-v1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = '1'
& $py -B tools/l1_rco_k1_folds_v1.py register --fold 1
& $py -B tools/l1_rco_k1_folds_v1.py prepare --fold 1
& $py -B tools/l1_rco_k1_folds_v1.py validate --fold 1
& $py -B tools/l1_rco_k1_folds_v1.py cpu-smoke --fold 1
& $py -B tools/l1_rco_k1_folds_v1.py evaluate --fold 1 --dry-run
& $py -B tools/l1_rco_k1_folds_v1.py admit --fold 1 --dry-run
```

All these actions hide CUDA internally. Default local cache is
`G:/temp_data/rsna-knee/l1rco-cache-full`. `cpu-smoke` actually backpropagates
through ConvNeXt on two real studies, changes an optimizer parameter, runs
holdout inference, and verifies exact interrupted loader/state replay with the
actual trainer and a tiny network. It is not a trained candidate or a GPU proof.
Admission prints `BLOCKED` until the reviewer provides the bound receipt.

The independent reviewer creates `TASK/f{k}/REVIEW.json` only after reviewing
the exact generated diff, sources and CPU evidence. Required fields:
`verdict: PASS`, distinct `builder` / `reviewer`, project-relative `report`,
`report_sha256`, and `bindings` equal to `k.review_binding(reg)` on that host.
Do not reuse Windows prepare/smoke hashes on Linux; paths and receipts differ.
This builder does not create a PASS review. Dry-run is not GPU authorization.

For local fold1 the exact default path is
`.local/scratch/astra-b2/h224-folds-v1/f1/REVIEW.json`. There is no required
schema/version field. After the independent reviewer issues an actual PASS,
B records it using this template (replace REPORT with that project-relative
PASS report; the prior FAIL report is not a PASS):

```python
from tools import l1_rco_k1_folds_v1 as k
k.configure(1)
reg = k.read(k.WORK / "registration.json")
REPORT = ".local/scratch/slot-b/k1-folds/ACTUAL-PASS-REPORT.md"
record = dict(verdict="PASS", builder="astra-b2", reviewer="astra-2",
              report=REPORT, report_sha256=k.sha(k.ROOT / REPORT),
              bindings=k.review_binding(reg))
# B only, after genuine independent PASS; never overwrite an existing receipt.
with k.REVIEW.open("x", encoding="utf-8") as stream:
    import json
    json.dump(record, stream, indent=2)
```

`bindings` must contain all six exact fields: `contract_id`,
`registration_sha256`, `primary_cohort_id_list_sha256`, `prepare_sha256`,
`cpu_smoke_sha256`, `fullcache_registration_sha256`. Gate compares the entire
object plus report SHA and distinct builder/reviewer. Re-run local
`admit --fold 1 --dry-run` after recording PASS. The example has not been
executed by the builder; no independent verdict is invented.


## Exact local GPU launch (B only; NOT executed by builder)

After independent review, a free GPU and CPU_ADMISSIBLE:

```powershell
& $py -B -c "import subprocess; from tools import l1_rco_k1_folds_v1 as k; subprocess.run(k.launch_command(fold=1), cwd=k.ROOT, check=True)"
```

`launch_command()` wraps `command()` with the canonical `tools/gpu_job.py` tag,
paths and process ownership. No hand-assembled child command and no `--wait`.
Exactly one local GPU job is allowed. Fit, final EMA export and full own-holdout
inference run under the same owner; collapse or incomplete output fails closed.

## Local hash export and portable bundle

```powershell
& $py -B tools/l1_rco_k1_foldcache_v1.py export-hashes --cache G:/temp_data/rsna-knee/l1rco-cache-full
& $py -B tools/l1_rco_k1_foldpack_v2.py build
& $py -B tools/l1_rco_k1_foldpack_v2.py verify --root "$task/package-v2"
```

`local32.json` selects eight studies from each hash fold with a fixed salted
ordering and hashes dtype/shape/pixels, metadata, normalized blocks and row
counts. Bundle: `TASK/h224-folds-v2.tar` plus `.tar.sha256`. It contains only
allowlisted source, frozen renderer tree, registrations, initialization/anatomy
weights and required derived training labels. It contains no raw DICOM/cache,
credentials, downloader or report text. Keep it private. The builder performs no
network transfer. B transfers the tar and its hash through the approved route.

## Linux setup and exact cache reconstruction (CPU)

B provisions a Linux instance with **one physical RTX4090**, sufficient CPU/RAM
and disk, Python3.12 plus matching torch/torchvision/timm/numpy/scipy/pandas,
pydicom/decoders, Pillow, safetensors and psutil. `environment-local.json` is the
installed-version reference, not a blindly installable Windows requirements file.
Package verification and real smoke detect missing dependencies. No installation
or credential handling is performed by these scripts. B/Taeyang supply the
original competition ZIP separately (already downloaded on the instance).

```bash
sha256sum -c h224-folds-v2.tar.sha256
tar -xf h224-folds-v2.tar
cd h224-folds-v2
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
python -B tools/l1_rco_k1_foldpack_v2.py verify
```

Provide the ZIP at
`storage/downloads/rsna-knee-abnormality-detection/rsna-knee-abnormality-detection.zip`
under this extracted project. Existing mounted data may be symlinked there by B.
Member names, CRCs, uncompressed/compressed sizes and series CSV SHA must match
the original cache registration. An arbitrary repacked ZIP is rejected.

```bash
python -B tools/l1_rco_k1_foldcache_v1.py render --smoke --workers 1
python -B tools/l1_rco_k1_foldcache_v1.py parity --smoke --cache .local/scratch/astra-b2/h224-folds-v1/render/smoke-cache
python -B tools/l1_rco_k1_foldcache_v1.py render --workers 6
python -B tools/l1_rco_k1_foldcache_v1.py parity
```

Original builder: `tools/l1_rco_fullcache_v110.py` v1.1.0, sealed registration at
`.local/scratch/l1-rco/fullcache/registration.json`. Reuses its Renderer without
changing calls/order: CPU anchored `v4`, size384, eval_stride1, three slices,
3.5 mm, no side model, two torch threads per worker, pinned anatomy model and
native geometry/pixel source. It rebuilds original train3190 + holdout977 in the
original order, independent of the new fold. Shards resume only under that exact
contract; assembly builds the common cache once for both folds. No CUDA is used.
Two-study smoke parity does not admit training: Linux admission requires all32
exact pixel/meta/block hashes and the completed full-cache receipt. Mismatch is
HOLD; do not relax tolerance or substitute a renderer.

## Linux fold preparation and GPU launch

After the full cache and32-study parity pass, run each CPU check serially:

```bash
for k in 2 3; do
  python -B tools/l1_rco_k1_folds_v1.py register --fold "$k"
  python -B tools/l1_rco_k1_folds_v1.py prepare --fold "$k"
  python -B tools/l1_rco_k1_folds_v1.py validate --fold "$k"
  python -B tools/l1_rco_k1_folds_v1.py cpu-smoke --fold "$k"
  python -B tools/l1_rco_k1_folds_v1.py admit --fold "$k" --dry-run
done
```

Evaluation references stay local, so no Linux `evaluate` command is required.
Obtain independent bound reviews for each host-local registration/prepare/smoke.
Before a new training host is accepted, B captures and compares runtime snapshots
against4070/Kaggle and records differences (no recipe mutation to force equality).
The `agents.md` platform rule additionally requires the known EXP127 AUG1 fold0
seed42 initial-state control against D-B-085 within4070 seed spread before trusting
the platform. That control is a separate B task, not implicitly passed here.

**B only**, after reviews, platform qualification and GPU0 idle:

```bash
# CPU-only plan check: no process dispatch, no GPU inspection.
python -B tools/l1_rco_k1_foldseq_v1.py launch-sequential --dry-run
# B only; keep this foreground process alive (e.g. B's persistent terminal).
python -B tools/l1_rco_k1_foldseq_v1.py launch-sequential
```

The equivalent canonical module entrypoint is
`python -B -c "import subprocess; from tools import l1_rco_k1_foldseq_v1 as q; subprocess.run(q.command(), cwd=q.ROOT, check=True)"`.
It uses the unchanged fold `launch_command()` / `command()`: f2/GPU0 first,
then f3/GPU0 only after f2's bound SUCCESS receipt. Both use the single
`TASK/gpu0-owner` scope and `CUDA_VISIBLE_DEVICES=0`. It also waits for f3.
The launcher binds run ID, command hash, wrapper/child identities and terminal
state through the canonical runtime. Failure, unknown status or interruption
stops sequencing without retry. `TASK/sequential-v1.json` records the attempt;
existing ledger/job directory/admission blocks relaunch, including successful
prior attempts. Do not delete evidence to retry. Interruption of this waiter
may leave the already-dispatched fold running; B checks that exact job before
any later action. No automatic cancellation or restart is performed.
Do not use the old `launch-pair` command (retained only to preserve code pins),
or concurrently launch any other owner on GPU0. OS/GPU/software provenance,
recipe, batch size and training timeouts remain unchanged. Actual launch is
Linux-only; the dry-run also works on Windows.

## Export, download, per-fold four-ruler evaluation

Production `run` automatically validates and exports each completed fold. B may
revalidate explicitly (never exports partial or smoke-trained models):

```bash
for k in 2 3; do
  python -B tools/l1_rco_k1_folds_v1.py export --fold "$k"
  (cd ".local/scratch/astra-b2/h224-folds-v1/f$k" && sha256sum -c SHA256SUMS)
done
tar -cf .local/scratch/astra-b2/h224-folds-v1/results-f23.tar \
  .local/scratch/astra-b2/h224-folds-v1/f2/EXPORT.json \
  .local/scratch/astra-b2/h224-folds-v1/f2/SHA256SUMS \
  .local/scratch/astra-b2/h224-folds-v1/f2/registration.json \
  .local/scratch/astra-b2/h224-folds-v1/f2/prepare.json \
  .local/scratch/astra-b2/h224-folds-v1/f2/fit/models \
  .local/scratch/astra-b2/h224-folds-v1/f2/fit/holdout.csv \
  .local/scratch/astra-b2/h224-folds-v1/f2/fit/result.json \
  .local/scratch/astra-b2/h224-folds-v1/f2/fit/fit-complete.json \
  .local/scratch/astra-b2/h224-folds-v1/f2/fit/runtime.json \
  .local/scratch/astra-b2/h224-folds-v1/f3/EXPORT.json \
  .local/scratch/astra-b2/h224-folds-v1/f3/SHA256SUMS \
  .local/scratch/astra-b2/h224-folds-v1/f3/registration.json \
  .local/scratch/astra-b2/h224-folds-v1/f3/prepare.json \
  .local/scratch/astra-b2/h224-folds-v1/f3/fit/models \
  .local/scratch/astra-b2/h224-folds-v1/f3/fit/holdout.csv \
  .local/scratch/astra-b2/h224-folds-v1/f3/fit/result.json \
  .local/scratch/astra-b2/h224-folds-v1/f3/fit/fit-complete.json \
  .local/scratch/astra-b2/h224-folds-v1/f3/fit/runtime.json
sha256sum .local/scratch/astra-b2/h224-folds-v1/results-f23.tar > .local/scratch/astra-b2/h224-folds-v1/results-f23.tar.sha256
```

B downloads the result archive and preserves its provenance in `TASK/downloaded`
(do not overwrite local prepare/review artifacts). Verify the archive hash and
each SHA256SUMS after extraction. Locally evaluate each own-fold CSV:

```powershell
& $py -B tools/l1_rco_k1_folds_v1.py evaluate --fold 1
& $py -B tools/l1_rco_k1_folds_v1.py evaluate --fold 2 --predictions "$task/downloaded/.local/scratch/astra-b2/h224-folds-v1/f2/fit/holdout.csv"
& $py -B tools/l1_rco_k1_folds_v1.py evaluate --fold 3 --predictions "$task/downloaded/.local/scratch/astra-b2/h224-folds-v1/f3/fit/holdout.csv"
```

Only aggregate results are printed: macro AUC over all12 targets for jev_hybrid
(binary), aquila_v1 (>=.5), dread (float32 rint), consensus (four-reference
agreement mask, >=.5). Uses original `exp156_k_sweep.references` and K1 AUC.
Exact own-holdout roster, finite probabilities, ok status and fold ID required;
an undefined target AUC is HOLD, never omitted. Original fold0 KEEP used exact976;
all977 and these other fold cohorts are descriptive comparisons, not the same
paired test. All outputs remain EXPOSED_EXPLORATORY / clean_oof=false. No gain,
ensemble decision or serving release follows automatically. Cross-host trajectories
cannot be paired to local controls; fixed-checkpoint/input inference parity on
the serving platform and independent result review remain B's gates.

## Capacity, estimates and limitations

- Cache upload366 GB at60 Mbit/s takes at least13.6 h excluding overhead; rebuild
  on instance avoids that transfer. The private source/weights bundle is much smaller.
- Full rebuild retains about366 GB of shards plus366 GB assembled arrays; reserve
  at least1 TB free **in addition to** the raw ZIP/any extracted data and OS/model
  allowance. Six workers need48 GiB concurrent disk margin and16 GiB available RAM
  at admission; provision32-64 GiB RAM or more and fast NVMe. Sequential fits share the cache and require capacity for only one training
  process at a time. Peak host RAM remains unmeasured; provision conservatively
  and check actual single-fit memory use. No two-GPU RAM estimate is implied. More VRAM does
  not permit a larger batch.
- Render time is unmeasured on the new instance: measure two-study smoke and use
  `4167 * seconds_per_study / workers` as an optimistic estimate; full I/O and study
  heterogeneity add overhead. Plan12-36 h CPU render provisionally, not a promise.
- At3.1 s/update the local fit is13.23 h plus inference.4090 speed is unmeasured;
  budget about27 h total for the sequential pair plus both inferences, after rendering. A faster GPU
  does not guarantee faster cache I/O. No training timeout is imposed.
- No live prices were queried. Illustrative single-GPU instance rate $0.35-0.60/hour
  and39-63 billed hours => $13.65-37.80 compute, plus storage/network/qualification.
  B checks actual quote, budget and reliability; this document authorizes no rental.
- Windows CPU checks do not prove Linux spawn/CUDA/driver compatibility. Linux
  smoke, exact cache parity, real GPU training and exports require B execution.
  See `TASK/BUILD-RESULTS.md` for actual builder evidence and unresolved blockers.
