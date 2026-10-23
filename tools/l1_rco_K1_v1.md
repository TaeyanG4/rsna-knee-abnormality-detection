# H-B-182 K1 v1.0.2 — single fold-0 RCO fit and evaluation

Changelog v1.0.2 / evaluator v1.1.0: B approved the exact Q12b976 primary comparison before candidate admission/output; retain all977 candidate predictions, register paired study jackknife uncertainty and signed directions, and bind independent review to the physical registration and primary cohort hashes. SHA-verified pre-amendment backups are in `.local/backups/l1-rco-k1-before-cohort976-v1.0.2/`. v1.0.1: validate both registered K0 evidence hashes before admission; mutation regressions exercise both inputs. v1.0.0: initial K1 launcher, durable resume, 977-study inference and CPU evaluator. The frozen parent, stage-0 sources, teacher and Q12b receipts are unchanged. Numerical/runtime recipe, queue tag and v1 filenames are unchanged. Derivative provenance is in `.local/scratch/l1-rco/k1/trainer.diff`.

## Registered experiment

One cold-start ConvNeXt-small spatial model, seed 0, E_prior, only fold 0, 16 epochs, 64 windows/study, batch 2, crop 352, original optimizer/loss/augmentation/EMA. The targets CSV contains exactly the 3,190 non-fold-0 training studies. The 977 fold-0 studies never enter the training pool, target rows, sampling or bias initialization. Bias is `logit(mean(train3190_only).clamp(.001,.999))`. The complete cache contains both rosters; only the separate, post-fit inference reads held-out pixels. Exposure remains EXPOSED_EXPLORATORY, not clean OOF.

The registration fixes the exact 977 inference roster and exact 976-study primary comparison, input hashes and evaluator before candidate output exists. The primary cohort is the ordered Q12b IDs, verified equal to sorted holdout977 intersect valid authenticated R7-f0 predictions. Its canonical JSON ID-list SHA256 is `fb04d36607a6fb9130357449dadb4f7c573183fdc033a1549975bd328a21942a`; Q12b registration SHA256 is `56d4338e00566f306694392cc7a4b2a72a9fc9133d7f15e2730f9096d7156a54`. KEEP requires the unweighted mean of the 12 target AUCs to satisfy RCO-f0 >= R7-f0 - 0.005 on at least 3 of the 4 fable rulers, AND the unweighted mean of 12 per-target Spearman correlations <= 0.93 on the 976. All 12 correlations must be defined. `tools/exp156_k_sweep.references()` is reused unchanged for JEV, aquila, Dread and consensus definitions/masks; average ties are used. No further cohort shrink, missing target or invalid prediction can yield KEEP. All977 valid RCO outputs remain mandatory; the excluded study cannot influence metrics.

**Excluded comparator study:** ID SHA256 `b85d664f1dbffca8259ed9240d0f106a4d80731eb8b913d1820559083fed7789`. Its existing teacher row has `status=dropped series` and fails the registered `status=ok` eligibility rule. Sanitized counts are input/rendered/used/dropped/skipped series 8/4/4/4/0 and DICOM/decoded/decode failures 260/260/0, with no recorded error. Root cause is unknown. The prior corrupt-archive/BadZipFile attribution was incorrect: this is not either receipt data-exclusion study. No raw ID or free-text error is emitted. After any candidate admission/output, registration amendments are refused, including checkpoint-only recovery state at `fit/ckpt/E_prior_s0_f0.pt` and when the original registration file is missing.

Per ruler, report the signed macro12 AUC delta, its direction even when the CI crosses zero, paired delete-one-study jackknife SE, normal 95% CI, and normal-approximation P(delta > 0). Delete the same study from both arms and every target, retaining original ruler masks; aggregate all12 target deltas before computing `SE=sqrt((n-1)/n * sum((leave_one_delta - mean(leave_one_delta))**2))`. CI is point delta +/- 1.959963984540054*SE and probability is Phi(delta/SE); zero SE gives 1/0.5/0 for positive/zero/negative delta. This deterministic calculation has no seed or bootstrap and preserves within-study target/model dependence. Studies are assumed independent. Undefined leave-one macro yields unavailable uncertainty without dropping targets. These are descriptive normal approximations, not posterior probabilities, and do not alter KEEP.

## CPU commands

Run from `H:/kaggle/competitions/rsna-knee-abnormality-detection` in PowerShell. These actions set `CUDA_VISIBLE_DEVICES` to empty internally. Do not rerun register/prepare after admission.

```powershell
$l1py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
& $l1py tools/l1_rco_k1_v1.py register
& $l1py tools/l1_rco_k1_v1.py validate
& $l1py tools/l1_rco_k1_v1.py prepare
& $l1py tools/l1_rco_k1_v1.py cpu-smoke
& $l1py tools/l1_rco_k1_v1.py admit --dry-run
& $l1py tools/l1_rco_k1_v1.py evaluate --dry-run
```

`admit --dry-run` reports BLOCKED until the exact independent review and completed full cache exist; it never launches or reserves a GPU. The full-cache gate checks `status.json` first and refuses a running/incomplete cache before inspecting payload metadata. Once complete, receipt/index/contract must agree on all 4,167 studies, sealed roster, contract, nonempty contiguous blocks, index hash, payload dtype/shape and committed file sizes. The reviewed fullcache builder's final receipt supplies whole-payload SHA256; admission deliberately does not reread the roughly 400 GB pixel payload.

Full relevant test suite, fixed basetemp, explicit CPU isolation:

```powershell
& $l1py -c "import os; os.environ['CUDA_VISIBLE_DEVICES']=''; import pytest; raise SystemExit(pytest.main(['-q','tools/test_l1_rco_v1.py','tools/test_l1_rco_v120.py','tools/test_l1_rco_fullcache_v110.py','tools/test_l1_rco_k1_v1.py','tools/test_l1_rco_k1_eval_v1.py','--basetemp=.local/pytest-tmp/l1-rco']))"
```

## Independent review receipt

A reviewer different from both builders must write `.local/scratch/l1-rco/k1/K1-REVIEW.json`, after reviewing the four K1 modules, two K1 test files, generated full trainer diff, this card and both build reports. It must contain:

```json
{
  "verdict": "PASS",
  "builder": "k1_builder + evaluator_parent",
  "reviewer": "distinct reviewer identity",
  "bindings": {
    "contract_id": "value returned by review_binding(reg)",
    "registration_sha256": "exact registration.json SHA256",
    "primary_cohort_id_list_sha256": "exact registered primary976 canonical ID-list SHA256",
    "prepare_sha256": "exact prepare.json SHA256",
    "cpu_smoke_sha256": "exact cpu-smoke.json SHA256",
    "fullcache_registration_sha256": "exact fullcache registration SHA256"
  },
  "report": ".local/scratch/l1-rco/k1/K1-REVIEW.md",
  "report_sha256": "exact independent review report SHA256"
}
```

The exact current bindings are in `BUILD-REPORT.md` and `admit-dry-run.json`. A builder must not generate a PASS receipt. The launcher rejects a missing review, changed receipt/source/artifact, or matching builder/reviewer. Independent result review is still required after training and evaluation before using a scientific decision.

## B-only launch

After the fullcache builder has exited cleanly, check its complete receipt, actual GPU processes and Shared Active compute; claim the local RTX 4070 Ti SUPER. Apply the independent review bound to the amended registration and rerun CPU admission. `CPU_ADMISSIBLE` is required. B then runs:

```powershell
& 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe' tools/gpu_queue.py .local/scratch/slot-b/l1rco/k1-gpu-queue.json
```

The queue contains one absolute-Windows-path entry, tag `L1_RCO_K1_F0_V1`. It starts `run`, which verifies live gpu_job parent/child identity, performs real admission, fits, then infers all 977 studies in the same resource claim. Direct `run` or `infer` is rejected. `gpu_job.py --list` must actually show the job running before B records launch. Training and inference logs are `fit/train.log` and `fit/infer.log` under the fixed work directory.

Estimated fit time on the 4070 Ti SUPER is **8.44 hours as a training lower bound**, from `3190 * 64 * 16 / 107.55145091156689 / 3600`. The estimate is printed before training. Full-cache I/O differences, checkpoints and final 977-study inference add time; total turnaround is not measured. This is not a training timeout. Measured per-invocation windows/s, elapsed training seconds and VRAM peak appear in `fit/status.json` and `fit/fit-complete.json`.

Admission requires the measured K0 peak plus current desktop usage plus 2 GiB <= total GPU memory, in addition to the reviewed <=12 GiB peak limit. Each training/evaluation heartbeat checks measured reserved memory against this gate. The outer watchdog preserves the reviewed 15-minute stale-progress and 10-minute WDDM paging checks, plus a 30-minute sustained throughput-collapse guard; normal checkpoint phases do not reset the collapse clock. Telemetry collection failure alone does not abort inference. A watchdog failure retains checkpoints and stops the verified child tree.

## Crash recovery — B only

Every 100 optimizer steps and at every epoch end, a flushed atomic checkpoint saves the model, optimizer, scheduler, EMA, epoch/batch cursor, epoch-start loader state, augmentation and CPU/CUDA RNG. At most 100 steps (less than one epoch) are lost after a crash. The last checkpoint is retained after fit completion. Resume recreates the same data permutation, skips committed batches and restores the saved randomness before the next optimizer step. Model/data/recipe/admission identity must match. A completed authenticated model bypasses fitting and reruns only inference if inference previously failed.

`gpu_queue.py` deliberately refuses a FAILED/orphaned/stale tag and never retries automatically. Re-running that queue is not a recovery command. B must first inspect job identity/logs, verify and clean up the old process tree, confirm no GPU job remains, retain/reclaim the Shared Active compute reservation, and confirm the original review/registration/cache still match. Then explicitly launch the same immutable command through gpu_job, which archives previous job receipts without deleting the fit checkpoint:

```powershell
$l1py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
$l1plan = Get-Content -LiteralPath '.local/scratch/slot-b/l1rco/k1-gpu-queue.json' -Raw | ConvertFrom-Json
$l1command = [string[]]$l1plan[0].command
& $l1py tools/gpu_job.py --tag L1_RCO_K1_F0_V1 -- @l1command
```

Never edit/forge old status or receipts, bypass a still-active resource lock, auto-retry, remove the checkpoint, or change batch/window/precision settings to recover. If the shared runtime still reports an uncertain active/orphaned identity, B must resolve that state through the existing runtime procedure before this command. No shared queue/runtime files were changed for K1.

## CPU evaluation after the completed job

```powershell
& 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe' tools/l1_rco_k1_v1.py evaluate
```

Outputs: `.local/scratch/l1-rco/k1/evaluate/result.json` (including four per-ruler uncertainty/direction summaries) and `per-target.csv` (48 aggregate rows). The evaluator authenticates candidate output via `fit/result.json` and the exact `holdout.csv` hash; CSV must contain only all 977 unique IDs, 12 finite probabilities, `status=ok` and `model_fold=0`, before selecting the registered976 for comparison. The real diagnostic dry-run computes all976 self-comparison and uncertainty with a separate 48-row table; it is never decision-eligible.

## K0 v1/v2 loss bound and limits

The retained rounded epoch-loss logs are v1 0.5819/0.5665 and v2 0.5818/0.5669. Allowing for four-decimal rounding bounds absolute differences by 0.0002 and 0.0005. The v1.2.0 derivative test mechanically proves its only trainer change is evaluation chunking; training code is unchanged. CUDA/cuDNN nondeterminism is plausible, not established causally: the missing per-step v1 record prevents an exact numerical replay. K1's fixed bias/roster correction is separate and intentional.

CPU qualification covers exact mid-epoch and epoch-boundary resume on real windows, full actual network backward, real holdout serving, target isolation, cache rejection and synthetic/real evaluator paths. Full-cache CUDA fit, CUDA resume and all-977 inference remain unexecuted here and belong to B's admitted job. No GPU, Kaggle, downloads, credentials or radiology text were accessed by this build.
