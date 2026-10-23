# OAI / RSNA differences and the next controlled K1 transfer

Recorded 2026-10-19 16:30 KST. B direct source/metadata review, not independent
review. This sharpens H216; it does not launch training or change a finalist.

## Decision

Use the same physically meaningful K1 input rules for both sources, complete
missing OAI metadata and anatomical views, then consider one small paired
OAI-adaptation/RSNA-recovery experiment. Do not directly train on the current
preview, repeat the previous mixed cold fit, or make OAI merely look like RSNA
with stronger smoothing or histogram matching.

The image pipelines can be aligned; acquisition contrasts, case mix and label
definitions cannot be made equivalent by resizing or normalizing intensity.
The old negative findings constrain confidence, not every possible use of OAI.

## What the existing preview actually shows

This comparison reuses the same exposed eight OAI and eight RSNA engineering
cases. It reads saved metadata only, not new pixels, labels or model outputs.
Cases are not a random estimate of either whole population, not paired patients,
and matching a plane does not establish matching pulse sequence or disease.

| Existing preview | OAI | RSNA |
|---|---:|---:|
| Studies / series | 8 / 32 | 8 / 46 |
| Sagittal / coronal / axial series | 8 / 16 / 8 | 18 / 17 / 11 |
| Native slice-center spacing, median (range), mm | 2.25 (1.50–3.00) | 3.30 (3.30–4.50) |
| Series averaged into thicker slabs by the existing K1 rule | 16 / 32 | 0 / 46 |
| Saved full-field physical extent, median (range), mm | 140 (140–160) | 160 (150–180) |
| Views per series in this prepared output | 1, full field only | 3 or 4, including anatomical views |
| Fat-suppression metadata known | 0 / 32 | 46 / 46 |

Full-field output dimensions are already384x384, with the same optional352
centre crop. The residual differences above are not evidence that one source is
intrinsically noisier or clinically poorer. Some visible smoothness can come
from the existing slab averaging. A single raw slice is not its slab average;
raw and processed display windows also differ in the viewer.

### Newly confirmed K1 input gap: fat-suppression metadata

The reused R6/R7 OAI extraction exports `StudyInstanceUID`, `SeriesInstanceUID`,
`Anatomical_Plane`, `Fluid_Sensitive`, but no `Fat_Suppression` column. Its frozen
route records omit that field as well. The native reader correctly represents
the absent field as unknown (-1); the OAI preview consequently uses K1 slot IDs
7/8/9 for8/16/8series. K1's training adapter includes the slot in each window's
metadata, and the model adds a learned slot embedding to image features. This
is a material difference in the complete model input, not only a display label.

This is a confirmed limitation of this new **preview-to-K1 bridge**. It does not
prove that OAI has no known fat suppression, that the pixel renderer is wrong,
that the old CoAtNet experiments had this exact defect, or that it caused their
losses. D238's full-field numerical/engineering result remains valid within its
scope; it was never a complete training cache.

Before any K1 training, bind the prior authenticated sequence/group provenance
to distinct fat-suppression and fluid-sensitive fields. Existing routing knows
SAG_IW_TSE, COR_IW_TSE and DESS water-excitation reformats. Do not infer contrast
from brightness, treat fluid sensitivity as fat suppression, or blindly copy
another reader's `-1 -> 1` assumption. Document verified mappings and keep truly
unknown cases explicit. The old metadata/source files and preview are immutable.

## Why the previous experiments did not work

1. **An early real input error:** D104/D116 established wrong-plane and
   wrong-contrast routing. The full header audit found960AX_MPR among2399knees
   routed to the sagittal non-fluid slot; other incorrect fills also occurred.
   Historical model exposure is not universally authenticated, and correction
   is not proof that every subsequent problem disappeared.
2. **A later corrected-input recipe genuinely lost:** D139's CoAtNet experiment
   had macro12 delta -0.0078 on the fixed held-out cohort. It replaced1048/3455
   RSNA draws (30.3%) per epoch with OAI and changed the BN population. The
   experiment supports rejecting that combined recipe; it does not isolate
   noise, OAI label quality, domain shift, or a causal OAI-only effect. It was
   not a K1 experiment.
3. **BN alone was not a sufficient repair:** D150's fixed-weight RSNA BN replay
   recovered about0.0005, leaving roughly -0.0073. This does not rule out every
   BN interaction. K1 uses a different architecture; no new BN replay is planned.
4. **External supervision is not automatically helpful:** D070/071/075's
   KneeMRI/MRNet auxiliary and rescue packages were negative. They were not OAI
   experiments, and their challenged causal explanations stay challenged.
   A smaller loss weight or a separate head by itself is not a new solution.

OAI native MOAKS subgrades and the competition's report-derived12targets also
have different meanings and missingness. Apparent absent diagnoses cannot be
filled with negative labels. OAI improvement alone is not RSNA improvement.

## Concrete sequence for H216

1. **Finish the input contract first.** Reuse the authenticated source groups,
   geometry and existing full-field result. Add a versioned metadata bridge,
   audit actual sequence semantics and unknown masks, then prepare only the
   missing K1v4 joint/patella anatomical views. Check physical crop bounds,
   source identity and the actual loader's slot/view/spacing/extent inputs.
   Same384pixels alone is insufficient. Preserve original RSNA preprocessing
   and all completed OAI preview artifacts; no blanket raw/cache rebuild.
2. **Preserve acquisition information.** Match by plane AND contrast family
   for visual/engineering checks. Use identical ordering/laterality/intensity
   rules where meaningful, physical scale and the existing slab policy. Do not
   fabricate a missing sequence from a different one. Do not add denoising,
   bias correction, first-echo selection or histogram matching without a
   specific demonstrated artifact and a separately fixed comparison.
3. **Freeze native supervision and separation.** Reuse authenticated participant
   exclusions. OAI's current preview cases are from an exposed evaluation set,
   not permission to place them in training. Define available structural native
   grades and masks before outcomes; prefer cartilage/meniscal structure with
   exact manuals and availability. Effusion/Hoffa-synovitis require separate
   semantic justification and are not automatic equivalents of RSNA labels.
   Unknown is masked. No invented ACL/MCL/fracture/contusion targets.
4. **One bounded pair, conditional on readiness/time.** Preserve immutable K1v3
   seed0. Start both arms from it with the same controlled parameter policy,
   RNG, windows, scheduler and EMA. The provisional budget remains400adaptation
   plus800RSNA-recovery updates per arm. Control:400RSNA continuation then800RSNA;
   candidate:400native-grade OAI adaptation then800identical RSNA recovery.
   Freeze early encoder layers, use/discard temporary adaptation heads, and
   restore the original12-target objective/head for the recovery phase. Exact
   tensor/head and optimizer contracts still require implementation checks.
   Original RSNA training history is retained in both. The control intentionally
   has400additional RSNA updates; this tests how to spend the same continuation
   budget, not an isolated OAI effect with identical RSNA exposure. No claim that
   all gradients, label semantics and exposure are matched.
5. **Judge on the target task before any expansion.** Compare both final arms
   and immutable seed0 on the fixed976/four report rulers using existing helpers.
   Report every target and uncertainty; rulers are correlated, not four
   independent replications. An arm that beats seed0 but loses to equal-budget
   RSNA continuation does not establish OAI benefit. Freeze the decision rule
   before launch; no best-epoch, target, mixture-weight or task search. Any
   retained candidate still needs an exact fixed-parent joint comparison and
   the actual seven-lens eligibility record. OAI-trained descendants do not get
   an independent OAI-panel certificate. Never add standalone AUC gains to
   forecast0.959/0.960.

Existing H210 keeps the sole GPU. H209/H211 priorities remain. H216 stays
conditional (confidence1, priority14), with missing metadata/full-v4/task/runtime
contracts. A rushed full OAI fit before the finals is not justified by these
eight cases. The appropriate next work is the concrete input bridge, followed
by the bounded pair only if its useful completion fits the remaining time.

## User extension: can OAI improve the existing L2 labels?

Yes as a research direction, but an OAI expert grade is a label for an OAI knee,
not a new ground-truth label for an RSNA knee. The missing bridge is both
**definition compatibility and reliable image-model transfer**. OAI alone does
not supply paired RSNA L2/true-label examples or a valid calibration of L2's
error rate on the RSNA population. OAI prevalence must not replace RSNA class
priors, and good calibration on OAI does not establish calibration on RSNA.

Current L2 is75%Dread +25%Aquila. D083 supports that historical minority-mix
advancement within its tested scope, not universal optimality. Aquila already
used images plus flight labels/report information. Simply increasing its share
or using another existing ensemble teacher repeats a heavily tested mechanism.

The most relevant counterevidence is D166: increasing Aquila to60% on five weak
targets improved the OAI lens while hurting RSNA/report and exposed Gold58
results (corrected Gold58 macro delta -0.001341, weak-five -0.001966). Therefore
"more consistent with OAI" cannot by itself mean "better RSNA labels". D180's
automatic report-silent-cell fill lost macro about0.0047 and harmed both image
lenses. D011 also changed prevalence-derived class weights along with labels;
future comparisons must keep original L2 class weights fixed to isolate the
selected intervention. D187/D191 preserve unsuccessful specific distillation
attempts, but their disputed explanations do not prove all label improvement
impossible. No old score was queried; only recorded discoveries were read.

### A defensible label-oriented route

1. **Define an allowlist before model output.** Start with native expert
   structural grades whose task definition and laterality/compartment can be
   authenticated. Cartilage damage is not automatically the competition's OA,
   and Hoffa-synovitis is not automatically its synovitis. Mark any unbridgeable
   target as ineligible for relabeling; it may still support auxiliary feature
   learning. Do not invent12OAI disease labels.
2. **Obtain evidence beyond the existing L2 teachers.** A teacher actually
   trained on correctly prepared OAI expert grades is a possible source. An
   unchanged R7/Aquila mixture is not new evidence. Public OAI/RSNA-trained
   weights need ancestry accounting; strong prediction is not independent truth.
   The H216 adapted model can be reused for descriptive disagreement only. Since
   it starts from K1 trained on the3190RSNA cases, its predictions on those cases
   are in-sample and must not be passed off as independent label verification.
   An OAI-only teacher with documented generic initialization, or participant-
   grouped inner cross-fitting for any RSNA-trained teacher, is needed for that
   stronger role. Extra cross-fitting cost is not silently authorized as ready.
3. **First produce a private disagreement audit, not an overwritten L2 file.**
   Restrict any proposed correction to RSNA training participants. Record the
   allowed target, frozen teacher/source provenance, uncertainty and domain
   support; no report text or label rows are printed. Disagreement identifies
   cases to investigate, not which source is correct. Teacher confidence alone,
   or an L2zero alone, never warrants changing a label or declaring it missing.
4. **Test only one selective intervention if the bridge is credible.** The first
   candidate could keep L2 values byte-identical and reduce loss weight only for
   a preregistered, supported uncertain subset. This tests confidence weighting,
   not corrected truth. If semantic support is insufficient even that subset
   remains HOLD. Generic disagreement weighting already failed its registered
   two-seed rule in D084 (H44/EXP150); a fresh name or an OAI mention is not
   sufficient novelty. Only demonstrably new expert-supervision evidence and
   a validated target/domain bridge justify reconsidering that mechanism.
   Keep original pos_weight, sampling, augmentations, updates,
   schedule, seed and evaluation fixed; compare against an unchanged-L2 arm.
   Do not simultaneously add OAI image training, change label values and change
   loss weights. Thresholds/caps must be fixed without reading outer outcomes;
   OAI-derived confidence thresholds do not automatically transfer to RSNA.
5. **Separate training utility from label correctness.** Use unchanged outer
   RSNA evaluation and full target-level harm reporting, with Gold58 descriptive
   only and source/exposure limits explicit. Do not evaluate against the new
   pseudo-labels, refit on the977outer cases, or claim four correlated rulers
   are independent clinical proof. In the absence of suitable independent RSNA
   expert labels, a positive training experiment may justify a candidate but
   cannot certify that individual L2 corrections are clinically true.

### Priority relative to H216

| Route | Present recommendation | Main missing evidence |
|---|---|---|
| Same input contract + staged OAI feature adaptation / RSNA recovery | Existing H216 first, conditional | Fat-suppression bridge, full-v4ROI, native task and bounded run contracts |
| OAI-informed selective L2 uncertainty audit / loss weighting | Design reserve; potential new expert signal | Compatible definitions, credible teacher transfer, ancestry/cross-fitting and RSNA support |
| Replace all L2, increase Aquila share, or fill all zero cells | Do not repeat | Prior adverse results; no new mechanism that resolves them |

This preserves a possible high-upside supervision path without pretending its
required teacher/target bridge exists. Do not create a second active GPU fit or
redo existing label/OOF/OAI-panel experiments merely for this proposal. A concrete
teacher and target contract would justify a separate candidate item; currently
the existing H216 input preparation is the common prerequisite. No original L2
row, weight, checkpoint, inference stage or validation target was changed.

## Evidence

- D104/116/139/144/150/070/071/075/238 and new D239; label-route evidence
  D011/083/084/124/130/166/180/187/191, including their later corrections.
- Private `oai-k1-preview8-v1/DOMAIN-SUMMARY.json` and
  `domain_summary_v1.py`: saved metadata, anonymous aggregates, zero pixel reads.
- `tools/r6_oai_readout_v110.py` prepare exporter and
  `tools/hb144_oai_public_v1.py` original route construction.
- Native reader: `.local/exp189-cpu-v120/input/cloud_code/training/native_cache.py`;
  slot IDs in `training/common.py`.
- K1 loader/head: `.local/scratch/l1-rco/k1v3-grouping/compact_train_l1.py`.

No new efficacy test, clinical image reading, label row output, model/checkpoint
load, GPU work, download, agent, submission or schedule was performed.

## 2026-10-19 16:49 KST: discoveries-first history and decision audit

The user requested another check against actual past discoveries before deciding
how to proceed. This audit read the relevant full entries, amendments and review
limitations, rather than relying on their original titles. It reuses saved
findings; no statistical analysis or completed experiment was repeated.

### An important correction to novelty

D133 and `.local/backups/replan-audit-20261012.md` explicitly record the earlier
H90 OAI-only/pretraining/mixed curricula and H100 OAI-teacher-to-label-correction-
to-student route. Both were DEFERRED, not tested negatives. Their blockers were
unvalidated intermediate stages, missing demonstrated transfer value and the
time before the freeze. The corrected CoAtNet result that followed was adverse
(D139), and the BN repair was insufficient (D150).

Thus H216 is a bounded, user-requested refinement of the H90 family, and the
label reserve revisits H100. Neither is a newly discovered general mechanism.
The new K1 input bridge and native-grade/staged specification are concrete work
differences, but fixing an input contract does not remove the efficacy or cost
blockers. Today's user authority permits useful preparation; it is not evidence
that those historical blockers have been resolved.

### Evidence that changes the decision

| Prior record | Result within its corrected scope | Decision for this continuation |
|---|---|---|
| D095 / D098 / D111 | L2 versus Dread improved five-fold JEV by about +0.0050 / +0.0054 / +0.0058 in RadR50 / ConvNeXt-T / DINO. Other rulers share ancestry; not independent clinical truth or automatic K1 transfer. | Keep current L2 as the unchanged baseline; it has replicated training-utility evidence. |
| D083 / D123 / D166 | Pure Aquila harmed its tested recipe; 50% did not broadly beat 25%; five-target 60% increased OAI agreement while RSNA/Gold evidence worsened. | Do not increase Aquila or replace all L2 as an OAI improvement. Historical gates and later corrections remain distinct. |
| D084 / D119 / D124 / D130 / D180 | Disagreement weighting failed its two-seed rule; silent-cell fill did not establish intended benefit, and R7 fill was harmful. | No generic disagreement mask, zero-to-positive fill or relabeled repeat. |
| D-A-006 / D-C-006 / D011 | Automatic correction lacked direct truth evidence; probabilities cannot reconstruct report missingness; changing labels also changed class weights in an old teacher test. | Require a new semantic/evidence bridge; keep original pos_weight and never infer missingness from an L2 value. |
| D104 / D116 / D144 | Real old routing defects were corrected; substantial processing/exclusion work already exists. | Reuse it. Only the missing K1 metadata/views/task binding is in scope. |
| D139 / D150 | The corrected OAI replacement recipe lost about 0.0078 macro; RSNA-only BN recovered about 0.0005. | No unchanged mix/refit or repeated BN rescue; no claim that denoising fixes its cause. |
| D070 / D071 / D075 | Other external-data auxiliary configurations and their rescue were negative; causes remain confounded. | A separate head, smaller weight or K1 architecture alone is insufficient novelty. |
| D133 | H90 and H100 were deferred design chains, not completed failed experiments. | Keep transfer conditional and the label-teacher chain out of the active GPU queue. |

### Chosen next step and explicit stop points

1. Complete a **missing-only CPU readiness note within H216**, using existing
   source/mapping helpers: authenticated sequence-specific fat suppression and
   fluid semantics; exact native-grade definitions/available masks; participant
   exclusions and ancestry; missing v4 view requirements. Show aggregates only.
   Preserve the completed preview and original L2. If a sequence or task cannot
   be authenticated, mark it unknown/ineligible rather than filling it by guess.
2. Use that note to decide whether the previously proposed **single staged
   transfer pair** is useful and affordable before the finals. Do not launch a
   teacher, student and label-correction chain. Keep the fixed same-start,
   400+800 updates per arm and unchanged RSNA L2 recovery as a prospective design;
   actual code, OAI throughput, preparation/evaluation time and current-job
   interruption/resume cost must be known before scheduling. These are real
   missing implementation facts, not a new long diagnostic framework. Do not
   stop H210 for CPU work or because a replacement merely has a plan entry.
3. A future positive transfer arm must beat the equal-budget RSNA continuation
   meaningfully in the full, fixed target/ruler accounting to support an OAI
   candidate. A positive comparison only against the original seed0 cannot
   establish OAI value. Mixed or adverse evidence is recorded honestly; no
   automatic epoch/target/ratio search or repeat fit. Original clinical-label
   correctness remains separate from training utility.
4. **OAI-informed L2 correction remains a reserve.** Reopening H100 requires a
   concrete new expert-derived signal, compatible target definitions and
   credible RSNA transfer/teacher ancestry. In-sample H216 predictions cannot
   certify L2 errors. Without those facts, preserve L2 and stop at a descriptive
   uncertainty audit; do not generate a replacement label file or a new fit.

H216 stays confidence1/priority14 and conditional; seven active items remain.
No new hypothesis ID, independent PASS, efficacy result, GPU claim, label change,
submission or interruption is created by this audit. H209/H211/H210 are retained.
