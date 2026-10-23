# Saved OOF error audit v2

`tools/oof_error_audit_v2.py` is the reusable improvement-loop error analysis tool.
It wraps the frozen `oof_error_audit.py`; that file and its EXP079/EXP081 consumers
are unchanged. This build is for B to test, obtain a **separate astra verification**,
and then run. Implementation is not an execution or scientific finding.

## Changelog

### 2.0.1 — 2026-10-10

- Fixes verification defect 1: the EXP175 helper binding was left at v1.0.0
  after that helper advanced to v1.0.1, so `load_inputs` rejected every run.
  During repair, concurrent edits advanced EXP175 to v1.1.0 and EXP177 to
  v1.0.1. Reviewed and pinned those current dependencies explicitly:
  EXP175 `8f1fee2acbbefdfba9c88651df4d54cf3360032c8cb865e6e042d109ca9e4460`,
  EXP177 `d65df611e76d06e8ae2fbd3d6e1c8edf271b7954117c8d3319231e637e4fcf67`.
  EXP175's `open_corpus` retains production contract, slot, ID, mask and shard
  checks. EXP177's reused budget, CPU limits, component draws, weighted AUC and
  interval functions retain this audit's contracts; neither helper's full audit
  entry point is called.
- Adds regression tests for every shipped helper pin, early rejection of a
  changed helper before loading data, and invalidation of a previous verification
  receipt when a helper is rebound. Existing tests and hash gates are retained.
- Bumps both tool and registry to 2.0.1. Requires a fresh separate verification
  receipt; no automatic hash refresh, input substitution or metric changes.

### 2.0.0 — 2026-10-10

- Replaces `.local/astra-improve/20261010-1329/build_error_package_v110.py`.
- Fixes the mislabeled ConvNeXt family: the registry binds **EXP170 L2**
  `outputs/results/EXP170_CONVNEXT_T_L2_S42_F*/oof_k36_flip_tta.csv` individually.
  It never calls EXP169 `paths()`, which points to EXP166/167 Dread ConvNeXt.
- Adds versioned data registry, exact prediction/summary hashes, identity checks,
  complete seed/fold grid, canonical held-out roster equality, and input rechecks.
- Preserves within-fold percentile rank, equal seed rank mean, global family rank,
  equal-three-family `bundle_of`, and global bundle rerank. This is the admission
  diagnostic policy; it does not establish serving aggregation/TTA parity.
- Adds all family/reference/target/fold AUC rows, including insufficient classes,
  masked denominators, full-cohort paired target and strict macro12 deltas,
  300-draw component intervals (seed 20261003), and v1 pair-error allocation checks.
- Keeps v1.1 strict rank thresholds while separating unknown reference agreement
  from observed disagreement. Adds middle-ranked negative confusion strata.
- Adds within-fold, exact contusion/effusion-matched MCL negative controls;
  retains unmatched cases. Randomly mixes images under neutral codes and keeps
  study IDs, strata and matching keys outside the blind image directory.
- Uses production `raptor_common.Corpus` through the existing validated cache
  adapter. All requested planes/contrasts are shown at 20/35/50/65/80 percent of
  filled slots, with zero-based absolute slot indices; missing groups remain blank.

## Reused implementation

| Helper | Use |
|---|---|
| `oof_error_audit.weighted_auc` | Tie-aware point AUC and metric cross-check |
| `oof_error_audit.target_metrics` | Existing dense-reference/fold metric implementation |
| `oof_error_audit.row_error_contributions` | Pair endpoint allocation; sum checked against `(1-AUC)/12` |
| `oof_error_audit.conditional_intervals` | Synthetic regression oracle for dense all-known references |
| `exp149_label_sources.fold_rosters` | Metadata-only canonical EXP099 cohort and held-out map |
| `exp115_vlm_eval.pct_rank` | Rank / n with average ties inside each fold and seed |
| `exp169_bundle_screen.load_oof/global_rank/bundle_of` | Exact CSV roster, global ranks, fixed bundle |
| `exp156_k_sweep.references` | Registered label rules, masks and minimum class sizes |
| `exp177_bundle_aggregation_audit` | Mask-aware vectorized weighted AUC, component bootstrap, intervals, CPU limits and budget |
| `exp175_image_domain_audit.open_corpus` | Existing production contract/ID/mask/shard checks around `raptor_common.Corpus` |
| `ext_zeroshot.groups` | Production slot group boundaries |

V1 `conditional_intervals` hardcodes `R1`, assumes all labels known, and does not
enforce reference-specific minimum class counts. Calling it on masked references
would change the estimand. V2 therefore uses EXP177's mask-aware vectorized form
of the same weighted statistic and tests equivalence to v1 on synthetic complete
data. `pfoa_complementarity_audit.pair_table` was inspected: its exact pair matrix
has hardcoded R1/ResNet keys and adds quadratic work unrelated to these requested
rank strata; v1's efficient endpoint allocation already supplies the needed check.
No report-audit tool is imported or executed.

## Registry and extension

`oof_error_audit_v2.registry.json` pins 25 OOF files plus run summaries:

- L2 RadR50: S42 EXP149 folds 0–1 / EXP157 folds 2–4; S43 EXP157 folds 0–4;
  plain `oof_k36.csv`, two within-fold ranked seeds averaged equally.
- DINO-base Dread: EXP160 BASE fold 0 and EXP164 folds 1–4; `oof_best.csv`, K36.
- ConvNeXt-T L2: EXP170 folds 0–4; `oof_k36_flip_tta.csv`.
- H1: EXP149 folds 0–1 / EXP153 folds 2–4, S42; plain `oof_k36.csv`.
- EXP172 DINO-base L2: explicitly **disabled** because all five registered output
  files were absent when built. No partial-fold or Dread fallback is possible.

EXP160 BASE's historical summary says internal fold 2. Its registry entry keeps
canonical fold 0 and binds arm/backbone/Gold exclusion instead of that misleading
internal field. `load_oof` must match every canonical fold-0 UID exactly; a mismatch
fails. The summary exception is recorded in the registry, not silently normalized.

To add EXP172 or H-B-85/86 arms, add/enable a variant with an explicit ID, family,
label recipe, prediction policy, path regex, expected seeds and **every** held-out
file path/hash. Bind summary hashes and stable identity fields (dotted keys work).
Never rename Dread predictions to an L2 identity. Add a named bundle mapping the
three registered roles `l2`, `dino`, `convnext` to desired variants. `bundle` remains
the primary stratum/montage diagnostic; additional named bundles receive metrics,
strata and paired comparisons. Additional standalone families require no code.
Arbitrary member counts/weight searches are intentionally outside this version.

Get a new independent verification receipt after any code or registry edit.
Missing/disabled variants are recorded, never automatically discovered or admitted.
The raw student manifest is read only as metadata; `_load_manifest` is not called.

For an early H-B-85/86 fold-0 arm, use a separately registered common-fold config:
declare `folds: [0]`, `expected_studies: 888`, retain **only fold-0 members of every
enabled variant**, and add the new matched arm. The entire canonical five-fold
split is still validated, but every metric, rank, mask, bootstrap and comparison
uses the declared common cohort. All participating variants must cover exactly
that seed/fold grid. Outputs record `analysis_folds`; a fold-0 result is not the
4,343-study pooled result. The shipped config remains five folds/4,343 studies;
missing EXP172 files cannot silently shrink it.

## Definitions and interpretation

For each target, JEV prevalence is computed on its **known** cells.
FN means known JEV positive with final rank `< 0.5`.
FP means known JEV negative with rank `> 1 - prevalence/2`.
Exactly-equal ranks are not errors. The ranks are the final global family/bundle
ranks after the registered within-fold construction, not calibrated probabilities.

Agreed positive/negative requires JEV, aquila and Dread all known and all equal.
Unknown agreement is reported separately. Disputed-error fractions and class
baselines use only cells for which all three labels are known; denominators are
written explicitly. Positive montage controls have rank `> .9`; low negative
confusion controls have rank `< .5`; stricter montage TN controls have rank `< .3`.
Middle-ranked negatives satisfy `.5 <= rank <= FP threshold`. These strata are
fixed diagnostic heuristics, not clinical sensitivity/specificity operating points.

Confusion tables cover MCL × contusion/effusion, Fracture × contusion, and
Lateral/PF OA × other compartments, on agreed-negative FP/low/middle strata.
Other-target rates respect that target's known mask. MCL matching uses JEV binary
contusion AND effusion and the same held-out fold, without replacement, from
agreed MCL negatives with rank `< .5`. Unknown confounders/no available exact
control are explicit unmatched rows, not relaxed matches.

Rescue counts use agreed-reference errors of A which B no longer flags. Alongside
them, paired AUC deltas are **B minus A on the full reference-masked cohort**.
Do not compute AUC on selected FN-only/FP-only cases or interpret rescue counts
as an oracle blend gain. Each draw resamples validation components with the same
multiplicities across families/references, holding ranks fixed. Missing component
membership and cross-fold components fail. Each reference retains its minimum
class count (consensus: 2 per class, others: 1). Macro12 requires all twelve targets
in both models and every included draw; invalid draw counts are retained. Sparse
targets/intervals, multiple comparisons, exposed development folds, shared weak
label ancestry, and unmeasured private distribution shift remain limitations.

## Outputs and blindness

All CSV rows/JSON files record version and the verification-note path/hash.
`blind/index.json` is the provenance sidecar for each PNG (including its SHA256);
the PNG itself has no label/stratum/UID metadata. PNG labels contain only a neutral
case code, slot group, fraction and slot index. Case filenames encode no stratum.
Cases are sampled with seed 20261003, deduplicated within anatomy, then separately
shuffled with seed 20261004 before neutral code assignment. Up to 8 per stratum,
both medial/lateral menisci, and matched controls for sampled MCL FP are retained.

- `auc_family_reference_target_fold.csv`: counts/AUC/deficit for every cell.
- `paired_auc_deltas.csv`: every ordered comparison, reference, target and macro12,
  95% paired component intervals and valid/invalid draw counts.
- `family_rescue.csv`: paired deltas plus agreed JEV/aquila/Dread rescue counts.
- `error_strata_by_target.csv`, `target_confusion.csv`: aggregate diagnostics.
- `allocated_pair_deficits.csv`: reused v1 endpoint-allocation conservation check.
- `strata_study_ids.json`, `rescue_study_ids.json`: study IDs for each stratum.
- `blind/panel_NNNN.png` and `blind/index.json`: blinded images and public index.
- `montage_key.json`: separate study/stratum key and all MCL matches/unmatched cases.
- `result.json`: written **last** only after input and verification rechecks.
  `RUNNING.json` is a start record; `FAILED.json` marks incomplete attempts.

Pixels come only from production `RAPTOR64_336_C140_S02`. Cache manifest/IDs/masks
and resulting PNG bytes are hashed; full memory-mapped image shards are not
rehashed (would threaten the CPU/I/O budget). Existing adapter checks layout and
shard coverage. Filled slots indicate nonzero cache images, not acquisition or
clinical coverage; neighboring cache slots are not guaranteed native neighbors.

## B's test, verification and execution commands

Run from the working folder in the existing Python environment with numpy,
pandas, scipy, scikit-learn, Pillow, threadpoolctl and pytest. The implementation
session did not run these tests or the audit. On this tool host the registered
Windows `py -3.12` launcher could not start its Python executable, and WSL syntax
checking was denied with `Wsl/Service/CreateInstance/E_ACCESSDENIED`. Consequently
even AST syntax checking is pending. B should use the project's working system
Python if this launcher remains unavailable.

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES=''
py -3.12 -m pytest -q tests/test_oof_error_audit_v2.py
```

After tests, dispatch a **separate read-only astra** pass against this spec and
the registry. It must check identity (including EXP170 and the EXP160 caveat),
hashes, aggregation, masks, minimum classes, duplicates, thresholds, bootstrap,
blinding and no report-text access. Have B record its PASS in a new JSON receipt,
for example `.local/astra-improve/oof-error-v2-verification.json`, with:

```json
{
  "status": "PASS",
  "tool_version": "2.0.1",
  "reviewer": "gpt-6-astra",
  "separate_pass": true,
  "review_note_path": "path/to/the/actual/read-only-review.md",
  "tests": "actual pytest result from B",
  "source_sha256": {"each path in SOURCE_FILES": "current full SHA256"},
  "registry_sha256": "current full registry SHA256"
}
```

This is a schema example, **not a verification receipt**. B can obtain the exact
source mapping without reading any data via:

```powershell
py -3.12 -c "import json; from tools.oof_error_audit_v2 import ROOT,source_hashes; print(json.dumps(source_hashes(ROOT),indent=2))"
Get-FileHash tools/oof_error_audit_v2.registry.json -Algorithm SHA256
```

Only after the separate PASS, B runs:

```powershell
py -3.12 -m tools.oof_error_audit_v2 --registry tools/oof_error_audit_v2.registry.json --output .local/astra-improve/oof-error-v2-run-001 --verification-note .local/astra-improve/oof-error-v2-verification.json --threads 1 --max-seconds 1800 --montage-per-stratum 8
```

Default one thread; CLI permits 1–4. Both wall and process CPU time have a
cooperative 1800-second ceiling checked between stages/targets/montages/hashes.
No subprocess workers, CUDA initialization or inference. A timeout leaves the
directory incomplete, so use a fresh output path for another authorized attempt.
Do not certify success from process exit without `result.json` status COMPLETE.
