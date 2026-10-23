# Expert MRI pilot96 preparation — v1.0.0

Registered 2026-10-19 before numerical case selection. Source: Taeyang's
instruction to proceed with the96-study pilot; parent protocol v1.1.0 in
`expert-mri-adjudication-plan-20261019.md`, D-B-235. B works solo.

## Scope and release state

Prepare a local provisional roster, blinded blank forms and an aggregate
receipt. No actual reading, new ground truth, MRI export, model fitting or
external reader contact is performed. Qualified MSK readers are not secured.
No AI reader substitutes for them. The old Aquila/L2 work is reused, not rerun.

The roster is **PROVISIONAL / READER HOLD**, not a certified clean clinical
panel. The existing component contract covers4343 non-Gold studies, including
all3190 K1 training and977 holdout studies, but not the58 Gold IDs themselves.
Direct Gold exclusion can be checked; patient linkage between Gold and the
new roster cannot be certified from this contract. Complete prior human-image
inspection history, patient linkage, viewer access and target-manual signoff
also remain unresolved. No images/headers/reports are scanned to fill gaps
beside active H210 training. Resolve these issues before issuing reading tasks;
if cases must change, version the roster and document exclusions without
looking at newly adjudicated outcomes.

## Fixed selection

- Seed20261019. One study per existing connected component, deterministic
  SHA256 ordering. Sampling is component-uniform, not study-uniform; record
  this population limitation and do not estimate whole-study prevalence from
  enriched strata. Select a hashed representative within each component.
- Reserve200 random K1 holdout components for a possible later test first,
  then32 random disjoint holdout components for pilot audit/development.
  The200 are only an exposed, conditional reserve; not a clean locked test.
  No test model/label readout occurs. Remaining holdout stays unused here.
- Draw16 calibration components from K1 training, then48 training cases.
  Calibration is excluded from subsequent matched training/evaluation. Any
  future fit needs a new roster/registration, not the old3190 contract.
- Training allocation18 random,15 shared contradictions,9 family disagreements,
  6 rare weak-label coverage. These are the full256 protocol proportions
  scaled exactly to48. Fixed precedence is that listed above. Missing strata
  use remaining random cases with explicit fallback counts.
- The eligible percentile pool is K1 training after calibration exclusion;
  all nonexcluded studies, not just selected cases. Average42/43 seeds within
  each family in float32. Treat ResNet50 and ConvNeXt-S as two families, not
  twenty independent predictors. Use saved H105 primary flip-mean OOF outputs
  pinned by its aligned input manifest; verify files, rosters, inventory and
  train/heldout provenance, including component disjointness. No model load,
  inference, metric/bootstrap recomputation or GPU access.
- Use the actual K1 target source, frozen L2=75%Dread+25%Aquila numeric columns.
  Never read report text or print rows. No weaker substituted reference.
- Percentiles use average-tie ranks divided by N. Shared contradiction: both
  families >=.8 and old target<=.2, or both<=.2 and old target>=.8. Alternate
  high/low queues for each target, round-robin over the fixed12-target order.
  Rank severity by min(percentiles) for high and1-max(percentiles) for low;
  ties by hash. These are suspicion rules, not diagnosed errors.
- Disagreement: descending absolute family percentile difference per target,
  round-robin with hash ties. Rare coverage: targets ordered by count of
  old L2>=.8 in the frozen selection pool, round-robin among those candidates.
  No claim of protocol/scanner coverage: suitable metadata was not inventoried.
- Each selected study will be read in full and for all12 findings. Target or
  stratum used to select it is private and absent from reader forms.

## Blinding and outputs

Private outputs stay under `.local/scratch/slot-b/expert-mri-pilot96-v1/`,
excluded from the team snapshot. A coordinator mapping holds real IDs, groups,
role and selection reasons. Reader templates contain opaque tokens, blank
finding/mapping fields and the fixed12-target list only. Separate randomized
orders for readers A/B; adjudication form separate. No patient-ID-bearing
viewer URLs, MRI files or derived labels are generated. Token pseudonymization
is not deidentification or approval for external transfer.

All determinate findings, read timestamps and locks remain blank. A filled
template is not imported as ground truth without actual qualified reading,
manual version, independent locks and adjudication. The whole pilot has no
training or efficacy endpoint yet. Proposed feasibility rules in the parent
protocol stay unchanged.

## Direct validation

Before selection, test deterministic group sampling, no cross-role/group
overlap, Gold exclusion, tied rank behavior, contradiction signs, target
round-robin/fallback behavior, no blank-to-negative conversion, and omission
of coordinator fields from reader views. Validate real saved output order,
finite12-target values, input hashes and eligibility counts. Check generated
counts, blank values and release HOLD again after writing. B direct tests and
review are not independent PASS. No existing research experiment is rerun.

Unresolved teacher/encoder ancestry and prior image exposure remain explicit.
Row-heldout disease-model predictions support suspicion sampling only; they
do not certify a nested teacher/student or final-blend evaluation.
