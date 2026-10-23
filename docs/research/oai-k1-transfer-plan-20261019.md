# OAI to K1: bounded input preparation and a prospective transfer test

Status: source/registration audit and design; no OAI-trained K1 exists from this work.
User requested active OAI use, preprocessing and inspection of all three folders.
B works alone; direct checks are not independent review.

## What is already known

- K1 v2/v3/v3-seed1 use the same 3,190 RSNA training studies, 977 held-out
  studies, L2 Dread75/Aquila25 targets and offline ImageNet22k-to1k ConvNeXt-small
  initialization. Actual v3 registration, full-cache registration and prepare
  command agree. No direct OAI image-training route is present. This does not
  certify the complete ancestry of the external anatomy anchor or all teachers.
- D104/D116 established old OAI plane/contrast routing problems. D144 confirms
  substantial corrected routing, sorting, physical crop and intensity processing
  already exists. Calling OAI wholly unprocessed would be wrong.
- D139 corrected-OAI CoAtNet lost macro12 about0.0078 in its tested recipe.
  It replaced1,048 of3,455 RSNA draws with OAI per epoch and changed the BN
  population. This is not a clean rejection of OAI or a K1 experiment.
- D150's BN-only replay recovered only about0.0005; do not repeat it.
- D070/071/075 already tested external auxiliary heads and a lower-weight,
  augmented, external-BN-frozen rescue, with adverse results. An auxiliary head
  or a new architecture name alone is not a novel justification.
- D144 authenticated1,504 historically eligible nonpanel OAI studies and557
  excluded panel participants. Counts cannot be transferred to a new native-grade
  task without checking masks. OAI-trained descendants do not become independent
  panel-eligible merely by masking targets or withholding some participants.

## Ranked directions

1. Match native OAI geometry and intensity to the actual K1 input contract,
   then test staged expert-supervised transfer followed by RSNA recovery.
   This has a specific difference from the old simultaneous replacement mix.
2. Investigate the already-downloaded T2-map first echo only if its acquisition
   geometry, contrast and usable information are confirmed. Historical consumed
   caches have zero such slots. Absence alone is not a harmful defect or a gain.
3. Reserve denoising/bias-field correction for a demonstrated artifact and a
   controlled comparison. Smoothness is not diagnostic fidelity. Thin-slice slab
   averaging already changes K1's noise/structure balance; do not stack smoothing
   methods or infer lesion preservation from appearance.

## Immediate bounded preprocessing: H215

Freeze8 hash-ordered studies from the existing extracted OAI confirmation461
roster before pixel reads; no labels or predictions select them. Reuse the prior
exact DICOM hashes/routes. These are visualization/engineering cases, not new
training examples or an untouched test. Reuse8 already-inspected RSNA cases
from D237 for a side-by-side reference, without repeating its audit.

Run one sequential CPU process with CUDA disabled, one numerical thread and no
checkpoint/model loading. Preserve original data. Reuse the exact K1 native
decoder, orientation/sorting, series-companded mapping, slab rule and full-field
384 renderer; expose the actual352 centre crop separately. No added denoising.
Only full-field views are rendered in this stage. SKM-TEA joint/patella ROI
anchors, the full v4 view set, training admission and clinical adequacy remain
unqualified. Never describe this output as a complete K1 training cache.

Write private `.npy` arrays and metadata plus anonymous PNG comparison sheets
under `.local/scratch/slot-b/oai-k1-preview8-v1/`. No MRI/UID/label data enter the
team snapshot. Original OAI DICOMs are opened read-only, never republished.
After completion open the actual existing RSNA cache, original OAI archive
directory and new OAI preprocessing directory in Explorer, as requested. Preview
images are display derivatives, not replacement source data.

Acceptance:8/8 cases, all registered series accounted for, finite correctly
shaped arrays, no missing/decode-error series, bound source/input hashes and
explicit slab-tail counts. Review all comparison sheets as engineering QA;
do not diagnose MRI pathology or create new truth labels.

## Prospective training design, not launch-ready

Preferred package: start from immutable completed K1-v3 seed0 weights, use a
short OAI phase to adapt the last encoder stage using separate native MOAKS
grade heads, then discard those heads and restore the original RSNA objective
in a fixed recovery phase. Keep the original3190 training/977 inference/976
comparison cohorts. Existing10-epoch K1 is never modified or extended in place.

Native cartilage extent/full-thickness subgrades and noncontrast synovitis
grades are candidate auxiliary tasks, not interchangeable competition labels.
Unknown values are masked, never negatives; ACL/MCL/fracture/contusion receive
no invented OAI targets. Exact column availability, definitions, participant
exclusion, task normalization and trainable parameter names must be bound
before any launch. No full clinical table or individual label rows are printed.

Two arms are necessary: an equal-update RSNA-only continuation control and the
OAI-then-RSNA package, both starting from the same fixed checkpoint. Proposed
bounded budget is400 adaptation steps plus800 identical RSNA-recovery steps
per arm, with the same batch/window count, seed, early-encoder freeze, scheduler
and final EMA. The control also uses/discards a temporary adaptation head.
These numbers are prospective design choices, not tuned or measured readiness.
They must be frozen with the actual implementation before results are read.
Both arms retain the entire original RSNA training history. The package changes
source and supervision semantics together; its result cannot isolate every cause.

Read every target/four rulers on the same976 and compare both with the original
seed0. Report all seven lenses or explicit ineligibility. Report-ruler agreement
alone cannot establish synovitis truth. Only a useful completed arm justifies a
fixed current-parent mixture check; no ratio/target/epoch search, automatic
submission or promised0.959/0.960 gain. Existing OAI557 inference is reused where
compatible and remains descriptive/ancestry-limited, not clean validation.

Current blockers: compatible full v4 OAI inputs/ROI, native-grade task binding,
actual bounded paired-job throughput/memory and time before the finals. Existing
H210 training continues and H209 preparation is not displaced. No future GPU
claim, full-cache rebuild, new download or paid resource is created by this plan.

## 2026-10-19 16:37 input and label-route clarification

D239 confirms that the reused OAI extraction has no Fat_Suppression column;
all32preview series carry unknown K1 slots. Preserve the completed pixel preview,
but resolve this metadata bridge and missing full-v4 views before any training.
The user also requested OAI-informed L2 improvement. Existing weighting/fill
negatives and the missing target/teacher/domain bridge keep it a design reserve;
no current L2 values or weights change. Full review and next steps:
`oai-rsna-domain-gap-review-20261019.md`.

## 2026-10-19 16:49 historical-predecessor clarification

The discoveries-first audit found H90 OAI curricula and H100 OAI teacher-label-
student designs explicitly deferred in D133. H216 refines the former; the label
reserve revisits the latter. These are not new general mechanisms or completed
negative experiments. Current user authority allows useful preparation, while
the old unvalidated-chain and cost blockers remain. See the16:49 evidence-to-
decision table in the domain review. Next OAI work is missing-only CPU readiness,
not an automatic fit or a reason to pause healthy training. Preserve original
L2, supported by D095/098/111, and the existing single paired-transfer design.
