# L1 RCO K0 V2 v1.2.0 — evaluation memory fix only

Changelog: retain v1.0.0 and its failed attempt; bound evaluation encoder calls
to 32 windows (was 512), empty unused CUDA allocator cache once on entering
evaluation. No memory-stat reset. Original no_grad, head pooling across ALL study
windows, training forward/gradients/optimizer/EMA/LR/loader/seed remain unchanged.
The existing forward pads encoder input to multiples of 32. No automatic shrink.
CPU bit equality cannot guarantee GPU bfloat16 bit equality or measured VRAM.

Data contract remains the sealed v1 registration (200 train +32 holdout). This is
intentional: the existing subset cache and fullcache v1.1.0 use that same data
identity. Evaluation source pins, trainer, proof and prepare hashes are bound
separately in the V2 independent launch review. No original file is edited;
failed K0 model/log/status/admission/failure are preserved and hash checked.

## CPU preparation (builder executes all)

```powershell
$l1py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
& $l1py tools/l1_rco_stage0_v120.py register
& $l1py tools/l1_rco_stage0_v120.py validate
& $l1py tools/l1_rco_stage0_v120.py prepare
& $l1py tools/l1_rco_runtime_v120.py cpu-smoke --cache .local/scratch/l1-rco/stage0/cpu-cache-reviewed
& $l1py tools/l1_rco_evalproof_v120.py
& $l1py tools/l1_rco_runtime_v120.py admit --dry-run
```

All CPU entrypoints set CUDA_VISIBLE_DEVICES to the empty string inside Python.
The eval proof loads only the preserved trained model and cached images, comparing
complete first two registered holdout studies at chunk 0 (unchunked), 512 and 32.
The CPU smoke covers real loader/geometry/cache parity, original parser/targets,
forward/backward and evaluation. No evaluation target labels are parsed.
Independent reviewer binds exact `bindings` from the dry-run in
`.local/scratch/l1-rco/k0-v2/K0-REVIEW.json`, using the v1 review schema; builder
then repeats the dry-run. Missing/stale review must block. The proof and prepare
hashes include the new source pins; never reuse the V1 launch review.

## B-only launch after the current 4070 job ends

Reconcile resource ownership and actual idle GPU before execution. New queue/tag
is L1_RCO_K0_V2; output is `.local/scratch/l1-rco/k0-v2/k0/`.

```powershell
& 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe' tools/gpu_queue.py .local/scratch/l1-rco/k0-v2/gpu-queue.json
```

The queue invokes gpu_job with tag L1_RCO_K0_V2 and this exact child argv:

```text
C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe H:\kaggle\competitions\rsna-knee-abnormality-detection\tools\l1_rco_runtime_v120.py run --cache G:\temp_data\rsna-knee\l1rco-cache --review H:\kaggle\competitions\rsna-knee-abnormality-detection\.local\scratch\l1-rco\k0-v2\K0-REVIEW.json
```

`prepare.json.argv` contains the complete unchanged training recipe with only
new trainer/output/targets paths. Exactly 200 training steps, batch2 x64 windows,
then all32 holdout studies. CT_PREDICT_CHUNK=32; the evaluation call also hardcodes
32 so an external environment cannot enlarge it. Original watchdog and owned
child cleanup remain. No additional training or automatic retry is authorized.

## Unchanged K0 and full-cache admission

Require 200 steps, finite/decreasing loss, >=80 real windows/s, finite completed
evaluation, max reserved <=12 GiB AND <=15e9 bytes AND desktop+peak+2GiB<=total.
Do not reset peak statistics between training and evaluation. V1's last emitted
12,501,123,072 bytes was a passing status, not the later failing peak: emit checks
the gate before writing. Full-stage memory remains UNMEASURED until V2 K0.

Fullcache v1.1.0 already accepts arbitrary in-project K0 paths and unchanged
numeric gates. Its separate admission review must hash the actual NEW result:
`.local/scratch/l1-rco/k0-v2/k0/result.json`. Use that path for `--k0-result` in
the existing fullcache launch card. Compatibility is tested without changing
v1.1.0 or manufacturing an actual K0 PASS. Full-cache admission remains blocked
until actual K0 PASS and independent result/admission review.
