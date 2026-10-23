# L1 RCO Stage 0 v1.0.0 — CPU preparation / B-only K0 card

Changelog: initial version. No original/frozen source or scored stage is edited.
Prerequisite independent pre-review: `.local/scratch/l1-review/stage0/PRE-REVIEW.md`.
Builder contract: `.local/astra-improve/BUILDER-CHECKLIST.md`.
Reviewer contract: `.local/astra-improve/VERIFY-RULES.md`.

## Registered scope and provenance

The exact public trainer is SHA256 `14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80`.
It does not implement mirror. Mirror is OFF; this is not a port of the separate
04df7 v9.8 trainer. Frozen originals remain untouched. `trainer.diff` lists every
change: zero bias, enforce single fold 0, 200-step cap, measurement hooks, bounded
checkpoint replace. ConvNeXt-S init, anatomy, renderer tree, px_cache, anchor bank,
roster/fold metadata and L2 targets are hash-checked. The CPU renderer derivative
uses the existing `exp189_s6_port._diagnostic_source` and no image-side model.

The cleaned full roster is 3190 train / 977 fold-0 holdout. Their UID hashes and
known connected-component hash are sealed. Gold is excluded. Every actual holdout
target is ignored before numeric parsing and replaced with 0.5 solely to satisfy
the original trainer's pool schema. Bias is zero; optimizer indices select only
nonzero folds. `--folds 4 --only-folds 0`; never `--holdout-fold 0` (that skips f0).
No full fit, full cache, serving release, download or Kaggle operation is authorized.

Exposure: **EXPOSED_EXPLORATORY**, `clean_oof=false`. This is known-component
isolation, not proof of independent anchor ancestry. SKM-TEA/Gold58 heuristic
tuning and non-independent label-derived rulers remain limitations. No OOF is
exported. The 200-step checkpoint is a partial exploratory smoke, not a finalist.

## CPU commands (builder runs these; B can repeat)

Run from the project root in PowerShell. Python commands set
`CUDA_VISIBLE_DEVICES=""` in-process before torch; PowerShell empty assignment is
not used. Scratch is fixed under `.local/scratch/l1-rco/stage0/`.

```powershell
$l1py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
& $l1py -m pytest tools/test_l1_rco_v1.py --basetemp=.local/pytest-tmp/l1-rco -q
& $l1py tools/l1_rco_stage0_v1.py register
& $l1py tools/l1_rco_stage0_v1.py validate
& $l1py tools/l1_rco_stage0_v1.py prepare
& $l1py tools/l1_rco_cache_v1.py --out .local/scratch/l1-rco/stage0/cpu-cache-reviewed --cpu-proof
& $l1py tools/l1_rco_runtime_v1.py cpu-smoke --cache .local/scratch/l1-rco/stage0/cpu-cache-reviewed
& $l1py tools/l1_rco_runtime_v1.py admit --dry-run
```

CPU proof uses two real train studies and one real holdout, original physical
window sampling/collation, full 352 spatial crop with two real windows through
the pinned ConvNeXt-S/spatial-attention model, backward and evaluation. It does
not qualify CUDA or full-batch memory. Dry-run is expected BLOCKED until the full
232-study subset and exact independent K0 review exist. GPU queue validation runs
inside `prepare`; it does not launch a job.

The CPU cache proof independently invokes CStudies serving preprocessing and
compares SHA256 of every physical eval window's pixels and metadata with the
assembled training cache loader. Every renderer crop and valid/padded area is
checked against source and output bounds. Receipts record the geometry coverage.
The earlier `cpu-cache` artifact is retained as superseded pre-review evidence.

## B: subset cache and independent admission

After Stage 0 implementation review, build only the registered 200 train + 32
holdout subset, serially on CPU (two torch threads):

```powershell
& $l1py tools/l1_rco_cache_v1.py --out G:/temp_data/rsna-knee/l1rco-cache
& $l1py tools/l1_rco_runtime_v1.py admit --dry-run
```

`receipt.json` reports bytes/study and seconds/study, checksum bindings, roster,
CPU threads and CUDA=false. Per-study shards plus assembled memmaps temporarily
consume approximately twice the final mmap size. A 4 GiB per-study margin and
assembly-size-plus-2 GiB gate protect disk space. No full-cache option exists.
Existing outputs must match the registration; partial shards resume only after
checksum checks. No user data is removed. The reusable study staging directory
is fixed and contains only this tool's generated extracted copies.

An independent reviewer checks the completed subset receipt and CPU smoke, exact
`prepare.json` argv and `trainer.diff`, then writes
`.local/scratch/l1-rco/stage0/K0-REVIEW.json` with this schema:

```json
{
  "verdict": "PASS",
  "builder": "l1-rco",
  "reviewer": "independent-lane-name",
  "report": ".local/scratch/l1-review/stage0/IMPLEMENTATION-REVIEW.md",
  "report_sha256": "actual report SHA256",
  "bindings": {
    "contract_id": "from admit-dry-run.json",
    "prepare_sha256": "from admit-dry-run.json",
    "cpu_smoke_sha256": "from admit-dry-run.json",
    "cache_receipt_sha256": "from admit-dry-run.json"
  }
}
```

These are placeholders, not an admission. The real runner refuses missing,
stale, self-authored or mismatched review evidence. No generic implementation
PASS substitutes for the exact completed subset receipt review. Before launching,
B checks Shared Active compute and actual GPU state, claims the exclusive local
4070 Ti SUPER, and executes the queue (which invokes gpu_job):

```powershell
& $l1py tools/gpu_queue.py .local/scratch/l1-rco/stage0/gpu-queue.json
```

The exact equivalent gpu_job command (use the queue above, not both) is:

```powershell
$l1plan = Get-Content .local/scratch/l1-rco/stage0/gpu-queue.json -Raw | ConvertFrom-Json
$l1command = @($l1plan[0].command)
& $l1py tools/gpu_job.py --tag L1_RCO_K0_V1 -- @l1command
```

`prepare.json.argv` is the complete trainer argv: 200 steps, batch 2, 64 windows,
384 render/352 crop, physical 3-slice 3.5mm, jitter .25, spatial attention,
gradient checkpointing, EMA .999, learning rates .0003/.001, weight decay .02,
16-epoch LR schedule, workers 6/3, eval stride 3 and predict chunk 512. The
16-epoch argument defines the smoke's subset LR schedule; both loops terminate
after exactly 200 optimizer steps, then evaluate all 32 holdout studies. It is
not a representative full-fit LR trajectory. Compile/mirror are off. No recipe
auto-shrink. Environment uses CUDA_VISIBLE_DEVICES=0, CT_PREDICT_CHUNK=512.

## K0 registered decision and projection

Before CUDA, capture actual desktop use/total from the only 4070 Ti SUPER and
refuse another Python ML process; desktop C+G apps are allowed. The runtime first
checks its own and parent's PID/create-time, running job state, exact command
digest and wrapper argv under the gpu_job claim lock. A direct invocation fails.
Only the verified wrapper's possible idle CUDA context is excluded from the
competing-process check; its memory is included in baseline desktop usage. Measure peak
reserved memory through training AND full holdout evaluation. Required:

- Exactly 200 optimizer steps and finite loss/logits; mean final 20 losses less
  than mean first 20 (no fitted-after-result criterion).
- At least 80 actual mask windows/s, loader-inclusive from first train phase
  through step 200, including warmup and epoch transition.
- Fable VRAM <=15 GB decimal **and** local peak reserved <=12 GiB **and**
  desktop-used-before + peak reserved +2 GiB <= actual card total.

The real required peak is UNMEASURED before K0. Fable's 15 GB alone is insufficient
for this desktop. Evaluation retains chunk 512; if that fails memory, stop and
register/review a separate changed recipe, never silently reduce it. GPU smoke
failure/partial attempts are retained; no automatic retry or training beyond K0.

Status heartbeat covers init/train/checkpoint/eval/complete; the parent reads
only status while running. Abort on 15 minutes without progress, the specified
WDDM high-util/low-power/near-full-memory signature plus 10 minutes without
progress, or 30 minutes below 25% of measured healthy commit rate. No fixed
wall-time training timeout. Owned child tree is stopped on failure.

`projection.json` keeps all full-cache, full-fit and T4 values UNMEASURED/null.
After K0, `projection-after-K0.json` provides provisional local estimates for
4167 cleaned studies: bytes/study ×4167, serial seconds/study ×4167/3600, and
3190×64×16 / measured windows/s /3600 (training lower bound; evaluation and
checkpoint overhead extra). The older ~390 GB claim is the unmeasured arithmetic
4407×606×384²=393.802 decimal GB, excluding metadata; it is not a capacity promise.
T4 serving increment stays **UNMEASURED even after K0** versus the approximate
2.2h reference headroom. Real T4 serving/parity qualification needs separate B
authorization. An independent result check and B decision are required before
any subsequent stage; passing K0 does not authorize a full cache or full fit.
