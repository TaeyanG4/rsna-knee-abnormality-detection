# H216 missing-input contract v1.0.0

Frozen 2026-10-19 before the new task coverage is counted. B solo, CPU only.
This refines deferred H90 (D133); it is not a new general transfer mechanism.

## Boundary and immutable inputs

Reuse the completed HB88 v2.0.4 header audit, authenticated by its existing
summary and output hashes. Do not scan DICOMs, rebuild historical caches, read
model checkpoints, inspect report/comment fields, change L2, or launch training.
All identities, task values and masks stay in private in-project scratch.
The new helper registers exact source/input hashes before execution, rejects
drift and existing attempts, and never changes the originals.

## Sequence bridge

Require the registered source sequence, plane, fluid flag, contrast family,
header-supported contrast certainty and method to agree. Map actual fat-signal
suppression to K1's binary Fat_Suppression field: FAT_SAT=1,
WATER_EXCITATION=1, NONE=0. Preserve the method separately: water excitation is
not TSE fat saturation and does not make DESS equivalent to TSE. This is an
explicit binary adapter definition, not an inference from brightness or the
Fluid_Sensitive flag. Preserve AX_MPR Fluid_Sensitive=-1. Unknown methods remain
unknown (-1) in the pure adapter and are not admitted in this registered cohort.
Contradictory registered routes fail closed; ambiguous split series exclude
the entire study. Use K1's exact pinned SLOTS table without model execution.

## Native tasks, frozen before counts

Use only baseline kMRI_SQ_MOAKS_BICL00, authenticated by the existing EXP136
receipt. Read numeric/identity/project/version columns through an allowlist;
never load narrative comments. Preserve the existing MOAKS project precedence;
no POMA, WORMS or binary proxy backfill. If equally ranked reads disagree on
the allowed task values, exclude the study rather than choosing the favorable
reading. Missing, unassessable and invalid values stay masked, never zero-filled.

The 30 temporary categorical tasks are the two documented cartilage subgrades
(any loss area, full-thickness loss area), each 0..3, for all14regions in the
existing CART_REGIONS, plus MEFFWK and MSYIC grades0..3. Cartilage code3.1 is
(3,1), not a continuous3.1 target. Reject invalid pairs such as1.2 and
follow-up-only0.5; require full-thickness grade <= any-loss grade. All tasks
use four-class categorical heads; no assumption that class spacing is linear.
Proposed loss: mean observed-task CE within each study, then mean studies;
skip all-masked studies. No prevalence weighting or target selection from results.

Definitions are authenticated local OAI documents:
- kMRI_SQ_MOAKS_BICL_Descrip.pdf pp2,4-6,8: centralized expert reads,14regions,
  compound cartilage coding, longitudinal reading and noncontrast limitations.
- kMRI_SQ_MOAKS_BICL00_Contents.pdf: exact baseline columns.
- kMRI_SQ_MOAKS_BICL00_Stats.pdf p38: effusion/Hoffa grades0..3 and missing codes.

MEFFWK can reflect both effusion and synovitis; MSYIC is noncontrast Hoffa signal.
Neither is declared identical to a competition label or clinical ground truth.
No ACL/MCL/fracture/contusion or meniscal tear labels are manufactured.

## Participant exclusion and next input sample

Exclude every knee/visit of every participant in the frozen557panel AND the
existing461R6 confirmation registration, plus the already-inspected preview
participants. Use the existing participant parser. Report the union count and
overlap, not participant rows. This preserves known evaluation exposure; it
does not establish unknown external model ancestry or a clean OAI test.
Retain studies with eligible unambiguous image groups and >=1observed task.
Select8 new eligible studies by SHA256('H216-NATIVE-PILOT-v1:'+UID), before
pixels or predictions, for a later missing-only full-v4 input pilot. That pilot
is not executed by this metadata tool and does not grant training admission.

## Acceptance and remaining gates

Source hashes unchanged, original files untouched, unique study/series keys,
participant overlap zero, exact tasks/masks and valid grades, same slot routing
as the pinned K1 table, no source/contrast conflicts. Tests cover each failure
mode rather than only successful schema construction. B's direct result check
is not independent PASS.

Full-v4 anatomical ROI preparation, actual paired trainer, named trainable
parameters, optimizer/EMA boundary, actual throughput and deadline feasibility
remain required. The prospective400+800updates per arm are unchanged. Equal
update budgets are not equal RSNA exposure; compare candidate with matched
control and original seed0. No new GPU claim, fit, label correction or release.
