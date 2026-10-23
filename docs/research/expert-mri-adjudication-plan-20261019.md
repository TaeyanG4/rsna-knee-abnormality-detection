# Expert MRI adjudication and matched label-correction study

Version 1.1.0 — 2026-10-19. Author: B / Codex. User: Taeyang.
Changelog: explicitly compare Opus/Astra-F/Aquila precedents and actual L2
student training; narrow novelty and require the existing strong label baseline.

Status: concrete proposed protocol, not an admitted experiment. Taeyang has
confirmed that no MSK radiologist is currently secured. This document completes
the requested planning work. It does not commission readers, disclose images,
create clinical labels, launch training or change the six active research items.
The existing local H210 training continues.

## 1. Decision and hypothesis

Test whether **new blinded expert image labels**, replacing erroneous training
targets on the same studies, improve a fixed MRI model and its fixed final
mixture. The first experiment changes supervision only. It does not combine
label correction with a new backbone, crop, loss, oversampling or augmentation.

There are three different phenomena to distinguish:

1. A model disagrees with an existing weak label. Either may be wrong.
2. Several distinct model families agree. They may all be right or all wrong.
3. Multiple models disagree with an adjudicated image label. Only now is a
   shared model error confirmed within the adjudication's uncertainty.

Do not call the selection list a list of known errors. The MRI readers establish
the new reference. Confirmed *label errors*, confirmed *model errors with correct
old labels*, and unresolved definition cases have different consequences. Only
the first produces a changed training label. The second is evidence for a future
representation/coverage hypothesis, not a reason to invent a replacement label.

D040 demonstrates shared errors among selected **report-label sets** on an
exposed small sample; it does not measure current K3/INF3 shared MRI errors.
D112 identifies diagnostic weaknesses of some report-derived references.
D103/D109 found harm from a package involving enriched Gold58 duplication and
changed update count; their accepted amendments do not isolate label-source
harm or establish equivalence. Therefore use new readings, preserve Gold58's
exclusion, and match images, sampling, masks and optimization between arms.
D235 is the motivating hypothesis, not evidence that this program will work.

### Direct precedent: this is an extension of Aquila, not a new general method

Taeyang correctly asks whether this repeats the Opus/Astra-F work. Broadly,
image-informed relabeling followed by student training **has already been done**.
Do not describe prior work as report-only, merely prompt experiments, or merely
prediction fusion. The new proposal is a human-adjudication extension whose
incremental benefit over the existing pipeline still has to be established.

| Dimension | Existing Opus / Astra F / Aquila | Proposed extension |
|---|---|---|
| Reader | General vision-language models; Opus pilot, Astra F production Aquila | Two qualified MSK readers plus adjudication |
| Evidence | F uses sampled15-slot image montages, flight_v4 labels, report text and task definitions; earlier B has no report text | Complete accessible MRI sequence review; first reads blinded to reports, weak labels, model scores and each other |
| Prior dependence | flight_v4 and report content can propagate into Aquila; dependence is documented | New image-only independent judgments reduce that particular anchoring route, but do not guarantee correctness |
| Training | Aquila actually trained students: 75%Dread/25%Aquila L2, tested across several architectures | Same-image/same-update matched current-label versus adjudicated-label study; no hard-case duplication |
| Selection | Broad production labels and exposed Gold58 reader/variant/fusion studies; uncertainty-targeting already simulated in D077 | Fixed random plus suspected-shared-error strata, with random confident-agreement cases and independent held-out image-label evaluation |
| What is new | Image relabeling, teacher blending and student learning are old | Actual qualified-human source, full-study inspection, blinded independent adjudication and a better-separated prospective evaluation contract |

Relevant actual student evidence, not leaderboard deltas:

- D083: pure Aquila loses JEV about0.0058 in the tested initial student screen;
  large target harms were retained. A minority mix worked better in that recipe.
- D095: L2 R50 five folds/two seeds JEV +0.004957, CI[+0.003782,+0.006200].
- D098: L2 ConvNeXt-T five folds JEV +0.005356 in the tested matched comparison.
- D111: L2 DINO-base five folds JEV about+0.0058; its registered paired primary
  effect was positive. These are report-derived proxy results, not independent
  human truth or private-score forecasts.
- D123/D166: increasing Aquila weight did not establish a general improvement;
  target/recipe harms remain. Do not repeat an Aquila-share sweep.
- D074/D077/D078/A051: reader agreement/fusion on exposed Gold58 does not isolate
  image contribution from prompt/report changes, and is not by itself student
  efficacy. Opus was not established to be useless: A051 reproduced better
  probability losses for Opus F despite the exploratory AUC/fusion ranking.

The recommended pilot therefore compares adjudicated decisions with **stored
Aquila as well as the current training labels** on identical cases, without
new Opus/Astra calls. Report shared expert/Aquila agreement and disagreement,
including uncertain cases. The main student control keeps the strongest
registered existing recipe, including any already-adopted Aquila contribution;
do not create an artificially weaker Dread-only control to make the new labels
look better. For K1, pin its actual existing target files rather than guessing
that its targets are the same as the older L2 recipe.

If human corrections disagree with Aquila but the fixed downstream comparison
does not improve, human superiority is not established. The full proposal
changes reader type, image coverage and blinding together; its result estimates
that package, not the separate causal effect of each ingredient. No full-MRI
advantage over the old montages is assumed in advance.

With **no human readers available**, rerunning Opus/Astra F under a new name is
not this proposed extension and is not admitted. A future AI-only variation
would need a specific prior failure it changes, a matched controlled test and
incremental evidence beyond existing Aquila. No such new run is queued here.

## 2. People and immediate deliverables

Recommended team:

| Role | Responsibility | Required availability |
|---|---|---|
| Two qualified MSK radiologists | Independently inspect complete permitted MRI series; enter all 12 target decisions and evidence | Both read every included study |
| Senior adjudicating radiologist | Resolve disagreements after an independent image assessment | About 20% of cases is a budgeting assumption, measured in pilot |
| B | Freeze splits/sampling, prepare access-controlled manifests and blank forms, authenticate predictions, implement matched experiment and analysis | No medical truth assignment |
| Data custodian / owner | Verify permitted reader access and approved viewer workflow | Before any study is shared |

If only one reader becomes available, the result is a single-reader feasibility
audit, not independent adjudicated ground truth. Two readers can discuss after
their original independent records are locked; unresolved cells remain unknown
if a third reader is unavailable. Never substitute a language model's answer for
the second radiologist or describe AI agreement as expert verification.

No paid or external contact is made by this plan. A future reader handoff needs
the named qualified reader, access arrangement, work scope, actual availability
and a quote. These are resource facts to obtain, not assumptions that an agent
can satisfy. Reader effort and currency cost are separated in section 9.

## 3. Cohorts, exclusions and leakage control

Before selecting suspicious cases, freeze connected patient/study groups using
the existing group contract. Include duplicate acquisitions and bilateral or
repeat examinations where the available identifiers support linkage; do not
pretend absent patient linkage is established. Exclude official Gold58, test
images, prior manually inspected study groups, and any prohibited source from
this new training/evaluation program. Record exposure and missing linkage.

Use four disjoint roles:

| Set | Initial full-study count | Selection / allowed use |
|---|---:|---|
| Reader calibration | 24 | Definition alignment before protocol lock; excluded from train/dev/test |
| Newly annotated training subset | 256 | Drawn only from the eligible training pool; balanced random and enriched sampling below |
| Image-label development | 120 | Group-disjoint random sample; used once for the prespecified development decision, not final claims |
| Locked image-label test | 200 | Group-disjoint random sample; sealed labels, never used for sampling, thresholds, epochs or mixture weights |
| Total | 600 | Design budget, not a statistical power guarantee |

The remaining eligible training studies retain their existing targets. The
256 cases replace labels **inside** that training pool; they are not added as
duplicate images. Calibration exclusions and any group exclusions apply to
both arms. The exact final N comes from a new split manifest; this must not be
silently launched under the old 3190-study registration.

The existing K1 977 holdout may supply a fresh *image-label* dev/test partition
only after group and ancestry checks. It is already exposed to prior model
selection, so it is not an entirely untouched dataset. Label freshness does
not erase image exposure. A stronger prospective claim requires eligible
previously unused groups or separately permitted new images; none are assumed
available now. If the current pool lacks enough eligible cases, mark the study
limited or blocked instead of moving training cases into evaluation.

Both matched study arms start from the same allowed pre-disease-training
initialization with known ancestry. Do not initialize from a checkpoint already
trained on the reserved test studies and then call that test independent.
Keep a provenance matrix for encoder pretraining, supervised fitting, teachers,
case selection and every model evaluated on each group.
Teacher-generated training targets must also exclude indirect supervised paths
from the reserved test groups; row-level OOF status alone does not certify a
nested teacher/student evaluation. Unresolved ancestry limits the claim to
exposed evidence rather than silently satisfying the eligibility check.

**Final-blend caveat:** the public/full-fold components of H189/K3/INF3 may have
trained on proposed evaluation images. Fresh reader labels do not fix this.
Full-blend independent gain can be claimed only with an all-member ancestry-
clean panel or authentic held-out outputs/retrained members for that panel.
Otherwise report the frozen-blend comparison as exposed diagnostic evidence,
separately from an eligible matched-model comparison. Do not silently add this
expensive retraining to the current queue.

## 4. Select suspected shared errors without circular reasoning

Use at least two distinct model families with authenticated held-out predictions
where available: for example existing R50-family and ConvNeXt-S MIL OOF outputs.
Count correlated seeds/variants as one family. R6/Raptor/K3/INF3 outputs are
eligible for this selection only if their actual training ancestry excludes the
case group. Do not use all-fold/full-data in-sample predictions as honest OOF.

Normalize each model's score to a target-specific percentile on its fixed
eligible training selection pool. This is a ranking device, not calibrated
disease probability. Freeze the pool, file hashes, family grouping and tie rule
before seeing new image labels. Raw 0.5 thresholds are not shared calibration.

The proposed 256-study training allocation is fixed as follows:

| Stratum | Studies | Purpose |
|---|---:|---|
| Random population coverage | 96 | Estimate ordinary correction yield; includes confident agreement with the old labels and cases every suspicion heuristic misses |
| Shared high-confidence contradiction | 80 | At least two eligible families in the top target-score quintile while old label is clearly negative, or bottom quintile while old label is clearly positive |
| Between-family disagreement | 48 | Large target percentile spread, selected by a fixed descending spread rule |
| Rare finding / protocol coverage | 32 | Predetermined underrepresented strata using permitted metadata and weak labels, expressly not assumed true positives |

For a weak soft label, provisional selection bands are <=0.2 and >=0.8; middle
values do not count as clear negatives/positives. These thresholds are selection
conventions, not clinical definitions. Use a deterministic seed 20261019,
group-level deduplication, fixed stratum precedence (random, contradiction,
disagreement, coverage), round-robin across 12 target queues, and a hash tie
break. A study enters once and is read for all 12 targets. If a stratum cannot
fill, fill the shortage from the remaining random pool and report the change;
do not relax thresholds after inspecting the new labels.

Distinguish high-confidence positive and negative contradictions, and preserve
the random subset. Looking only at disagreements would miss cases where every
model and every old label are wrong in the same direction. Do not infer
population error prevalence from the enriched 160 studies; report that only
from the random subset, with intervals and its finite coverage.

Keep the selection reason/model predictions outside the reader view. Confirmed
shared-error rates are calculated only after independent readings are locked.
Missing OOF inputs are a preparation dependency, not authorization for new
large inference or a reason to substitute leaky predictions.

## 5. What the radiologists read and record

Use the complete permitted study in a DICOM viewer with correct physical slice
order, orientation, target knee and sequence information. Permit scrolling,
window adjustment and multiplanar/sequence comparison; thumbnails or a few
model-selected slices are insufficient for an adjudicated negative. A missing
or degraded necessary sequence is recorded as not assessable, not normal.

Readers are blinded to reports, old labels, model scores, heatmaps, stratum and
the other reader's result during the first pass. This project does not give B
permission to access report text or export raw MRI. Use controlled access with
no external data upload unless separately permitted.

The target manual needs a versioned sign-off by the lead MSK reader. Use the
official competition definitions and retained primary host sources as the
starting point, not general diagnostic intuition. D013's accepted scope records
high-grade ACL/MCL and moderate-large effusion/Baker's conventions; confirm the
exact current wording before use. Do not invent a numeric grade threshold where
the retained source is incomplete.

| Targets | Required boundary to resolve in the manual |
|---|---|
| ACL, MCL | High-grade versus low-grade/indeterminate injury; chronic or postoperative cases |
| Medial, lateral meniscus | Tear versus intrameniscal signal/degeneration; root and postoperative findings |
| Medial, lateral, patellofemoral OA | Compartment, cartilage/severity boundary and required structural evidence |
| Effusion, Baker's cyst | Official moderate-large threshold versus trace/small; cyst presence alone is insufficient |
| Synovitis | Direct evidence and sequence limitations; effusion is not an automatic surrogate label |
| Contusion, fracture | Bone-marrow edema versus contusion versus fracture; separate evidence for each target |

For each study-target cell record: present / absent / indeterminate /
not-assessable; structured severity/definition code; evidence series and
slice span or anatomical location; confidence high/medium/low; reason code.
Confidence is an assessment flag, not a disease probability or automatic loss
weight. Any optional narrative must remain a private annotation artifact and
must never contain copied clinical report text or identifiers.

Both readers independently lock all 12 cells. A third reader first records
their own image assessment, then reviews disagreement reasons and issues the
adjudicated decision. Preserve both original reads and the adjudication;
never erase disagreements. A conflict is not resolved by majority model vote.
If the target definition is unresolved, retain the ordinal finding and mark
binary mapping unresolved. A later manual change requires a new version and
consistent re-adjudication, not selective changes to improve a metric.

Quality controls: a fixed 10% blinded duplicate reread after an appropriate
washout, completeness checks, positive/negative agreement, per-target
disagreement and not-assessable rates, and kappa with prevalence caveats. Kappa
alone is not a truth certificate. Use the first calibration set to measure
actual reading time and fix definitions before production labels are collected.

## 6. Pilot: 96 studies before committing to 600

With two secured readers and an adjudication route:

- 16 calibration cases; excluded from all later training/evaluation.
- 48 suspected-error/coverage training-pool cases, drawn by the fixed logic.
- 32 random audit cases, separate from the final test; no final efficacy claim.

Pilot cases can be counted toward the later corresponding calibration/training
budget after protocol freeze; the32 exposed audit cases count only toward the
120-case development set and cannot become the locked test. Thus the base
budget remains600 distinct studies if all pilot cases are eligible. If the
manual changes, re-read affected pilot cases consistently or disclose the extra
excluded-case cost. Do not select pilot cases using future test predictions/labels.

Pilot deliverables are reader time, definition resolution, inter-reader
agreement, confirmed old-label corrections, confirmed shared model errors,
assessability and corrections per reader-hour. No model training is needed to
answer those initial operational questions. A few corrected examples do not
demonstrate that a model can learn a useful complementary signal.

Proposed operational progression rule, locked before pilot reads: proceed to
broader annotation only if (a) an adjudication path and all intended target
definitions are workable, (b) at least 80% of attempted target cells are
assessable and at most 10% remain unresolved after adjudication, (c) at least
20 confident changed training cells spanning at least 10 studies occur in the
48 training-pool cases, and (d) the measured reader effort fits an actual agreed
budget. These are pragmatic feasibility gates, not validated efficacy
thresholds. Failure to meet them leads to a definition/access redesign or a
parked project, not relabeling uncertain cells to pass. If only a subset of
targets is workable, a separately versioned narrower pilot is required.

If most new labels agree with the old ones while models remain jointly wrong,
record evidence against the label-noise hypothesis and stop the label-repair
training branch. Investigate image coverage/representation only through a new,
nonduplicate proposal. Do not add a crop experiment to this one.

## 7. Train a controlled pair, then replicate if warranted

Use the K1 v3-G architecture and its established 12-target recipe as the first
student. Build a new registered study wrapper; do not modify live K1/H210 or
submitted stages. Pin image renderer, initialization, seed, cohort, batch order,
optimizer grouping, losses, LR1e-4/head3e-4, warmup, ten epochs and final EMA.
No H211 augmentation change is combined with this treatment.

| Arm | Inputs and learning target |
|---|---|
| C: matched original-label control | Same eligible training images and target masks; original targets in all usable cells |
| E: expert-corrected labels | Same images/order/updates/masks; substitute adjudicated 0/1 only in determinate reviewed cells, retain original targets elsewhere |
| Existing production reference | Frozen historical model for context only; not a substitute for C unless every lineage/data/mask/update condition is demonstrably identical |

For a reviewed unresolved/not-assessable cell set the training loss mask to zero
in **both C and E**. Nonreviewed usable cells keep the same original target and
mask in both. Preserve any source missingness policy. This prevents the effect
of masking from being mistaken for the effect of changed answers. The pair's
control may therefore differ from the historical production recipe; report it.
Use equal per-cell weight on included cells and the same batch denominator.
Do not turn radiologist confidence into an uncalibrated soft label or upweight
the reviewed subset. The existing loss/mask adapter must be smoke-tested before
training; it is proposed, not already implemented.

Every study remains sampled once under the same fixed sampler. No Gold58,
four-copy enrichment, new class weights, longer schedule, manual hard-example
duplication or new loss. Export the final EMA, not a selected best test epoch.

First pair: seed0 C and E. Repeat a fixed seed1 pair only if the development
rule supports continuing. Evaluate all promised seeds, including adverse ones;
do not choose the better seed after viewing the locked test. A versioned
manifest binds cell counts/changed-mask counts and label hashes without printing
patient IDs or label rows in shared notes. New labels stay in a private approved
annotation store, not the team snapshot, public dataset or .local/relabel.

GPU planning baseline: the current K1 recipe has taken roughly 13–16 hours per
fit. A matched two-fit screen therefore budgets 26–32 local GPU-hours plus
inference; a second paired seed makes 52–64 hours plus inference. These are
borrowed planning ranges, not measured timings of a new split/mask adapter.
One local GPU job only; neither current H210 nor a qualified candidate is
interrupted. Actual free-slot admission and a scoped CPU/real-environment check
precede any future launch.

## 8. Evaluate learning, then contribution to the final system

Primary study endpoint: expert-label macro AUC across the same fixed 12 targets
on eligible held-out image-label data. Report each target, the original weak
four, and separate known shared-error strata. Include correction-subgroup
performance only as a secondary diagnostic; it must not replace population
evaluation. Report old weak-ruler metrics as diagnostics, especially where
their definitions disagree, not as the sole truth of this intervention.

Development decision (proposed, not retrospectively tuned): after the seed0
pair, continue to seed1 only if mean macro delta is positive, no target point
delta is below -0.01, no material execution/label-definition defect remains,
and the available class counts support interpretation. The target floor is a
coarse operational screen, not a proof of clinical safety or noninferiority.
Freeze the two-seed averaging rule before opening the test.

Locked test is opened once after training/selection are frozen. Use the same
patient-component bootstrap draws for C and E and account for repeated exams;
report paired intervals and both seed effects. Image-label reader uncertainty
is additional to sampling uncertainty: repeat the readout against each locked
reader's labels as a sensitivity analysis and preserve disagreement masks.
Do not hide targets with insufficient support using nanmean.

Planning minimum: seek at least 20 positives and 20 negatives per target in the
locked test, explicitly only a support floor. A 200-case panel may not reach it
for rare conditions and cannot reliably certify a +0.001 AUC gain. Before any
model-test score is unblinded, use blinded adjudicated class counts to decide
whether to add the next two predetermined random batches of100 (cap400 test;
up to800 total readings). No outcome-based optional stopping. Still insufficient
support => incomplete twelve-target conclusion, not an automatically reduced
macro. Estimate the precision/cost needed for a meaningful minimum effect from
development data and the paired design; do not promise statistical power from
the nominal600-study budget.

For confirmatory wording, require positive held-out paired macro effect with a
95% interval excluding zero, consistent seed direction, and all adverse target
outcomes shown. Otherwise call it promising/inconclusive or adverse as warranted;
do not convert a wide interval to equivalence. Twelve secondary target tests
need multiplicity-aware interpretation. Even a positive study does not estimate
the probability of a particular Kaggle score.

Only then assess one separately registered fixed production mixture: replace
the existing RCO slot with the corrected model at the existing90:10 operator.
Compare it with the same mixture using the matched control and the retained
production reference on the same eligible studies. Freeze the exact parent
version before analysis; no target-specific weights or alpha sweep. Apply the
all-member ancestry caveat above. Standalone improvement or lower error
correlation is not proof of ensemble improvement.

Any future release still needs working repaired serving, explicit RCO COMPLETE,
all IDs x12 finite, material failure/codec checks, actual runtime evidence,
fresh quota/duplicate check and canonical submit-once. This protocol grants no
submission, new Kaggle training or automatic retry.

## 9. Resource budget and realistic calendar

Let N be distinct studies, m the average minutes per independent full study
read, d the adjudication fraction, a adjudication minutes and r=0.10 blinded
reread fraction. Budget total reader-hours as:

`2*N*m*(1+r)/60 + d*N*a/60`, plus onboarding/manual/viewer time.

Using **assumptions**, not observations, m=8–12, d=0.20 and a=10–15:

| Package | Reader-hours including rereads/adjudication | Additional setup | Meaning |
|---|---:|---:|---|
| 96-case pilot | about31–47 | 6–10 hours | Feasibility and error yield; no small-gain validation |
| 600-case initial study | 196–294 | 8–12 hours | Two-read image labels; rare-target support may still be inadequate |
| 800-case expanded study | about261–392 | Same setup if definitions unchanged | Optional blind test-support expansion, not guaranteed power |

Replace these numbers with measured pilot medians/upper-tail times. Two readers
each available six hours/day provide twelve first/second-read hours/day, not
twenty-four. The initial600 study requires roughly15–22 reader working days
for the two-reader/reread portion, with a third reader's20–30 adjudication
hours partly parallel. With coordination and QA, plan roughly3–5 working weeks
after readers/access are secured. Quotes are unknown: cost = actual contracted
hours times each reader's rate, plus any approved viewer/custodian cost. No
currency total or staffing availability is invented.

Relative calendar:

1. D0: both readers/access confirmed; freeze scope, groups and reader manual.
2. D1–D4: calibration and96-case pilot; measure time and apply feasibility rule.
3. D5 onward: production reads in fixed batches; seal each batch, report only
   counts/quality to the training coordinator while locked-test labels stay closed.
4. After training-label freeze: seed0 matched GPU pair, development readout;
   if warranted seed1 pair, then one locked-test analysis and one fixed joint.

As of2026-10-19, readers are **not secured**. The final-pair deadline is10-22
and competition ends10-23 08:59 KST. A600-case independently adjudicated,
matched/replicated study is not a credible baseline plan for this deadline.
Even immediate recruitment makes the96-case pilot a tight feasibility exercise,
not a justified0.959/0.960 model. Keep current candidate preparation moving;
do not consume the remaining deadline by pretending this new study is ready.
An accelerated route needs actual staffing/time evidence and a separate scope
decision; no schedule or GPU reservation is created here.

## 10. Deliverables and failure decisions

Planned artifacts, created only after the prerequisites are met:

- Versioned target manual and a blank reader form (companion document available).
- Private group/role/selection manifest with exposure and model-ancestry matrix.
- Locked independent reader records, adjudication records, immutable label
  versions and aggregate QA/correction-yield report.
- Matched C/E training registration and source/label/mask hashes, two-seed
  outputs if warranted, fixed statistical report with all harms and uncertainty.
- One exact-mixture report and, only if warranted, a new immutable release stage.

Stop or narrow the claim when readers/access are unavailable, definitions remain
unresolved, correction yield is small, models rather than labels are wrong,
test ancestry is ineligible, rare-target support is absent, matched treatment
hurts, or deadline/budget is insufficient. Do not solve these failures by using
AI pseudo-labels as truth, recycling Gold58, dropping adverse targets or tuning
the ensemble on the test.

Current next action is resource/design feasibility, not GPU admission. D235 is
updated with this plan and the user's no-reader status. No new active candidate
or scientific result is recorded. B's design review is not independent review.
