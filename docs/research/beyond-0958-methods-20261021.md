# Methods worth reviewing beyond the reported 0.958

Version 1.0.0 — 2026-10-21 KST. Author: Codex, slot A, for Taeyang.
Research record: D-A-052. Status: **PENDING independent review; no efficacy experiment and no execution admission.**

## Recommendation

There is a concrete new experiment worth reviewing: **train K1 against one complete study-level attention loss using encoder-gradient replay**, instead of assigning the study label to a randomly capped 64-window bag. This changes the supervised objective while keeping the deployed architecture fixed. It is more specific than the already unsuccessful coverage-preserving sampler proposals.

For a larger representation change, **knee-specific self-supervised pretraining** deserves a second, longer-horizon investigation. Newly located primary sources provide knee-MRI transfer evidence, although they do not establish improvement over our K1 or R7. This develops the existing D-B-235 reserve; it is not a newly invented mechanism.

For better labels, the strongest credible route remains **new, blinded, complete-MRI expert adjudication**, already designed in D-B-235. Reuse its preparation. No newly located public label release establishes a better independent twelve-target truth source. Its missing reader/access resources remain real even when the deadline is ignored.

The user's 0.958 and leader's 0.969 are reported reference points, not evidence about the leader's technique or a measured 0.011 opportunity attributable to one cause. None of these proposals has demonstrated a Kaggle gain. Deadline proximity was deliberately excluded from their scientific merit; costs and unresolved resources are still stated.

| Review order | Proposal | What changes | Novelty relative to this project | Present readiness |
|---|---|---|---|---|
| 1 | Full-study loss with gradient replay | The entire selected study bag participates in a single attention normalization and loss; encoder stays trainable | No completed full-bag replay experiment located; overlaps the coverage question but tests a different objective | Concrete controlled design; implementation, numerical parity and resource review still required |
| 2 | Knee-specific masked MRI pretraining | Learn anatomy/appearance without disease probabilities, then supervised fine-tune | D-B-235 extension with new knee-specific literature support | Research reserve; exact architecture/checkpoint availability and cost unresolved |
| Label route | Blinded expert MRI labels | Acquire new image evidence for shared label errors | Existing D-B-235/H213 preparation, not a new experiment proposal | Reader/access/ancestry HOLD; do not repeat selection or use blank forms as labels |

No active B item is claimed, rescored or superseded. In particular, this does not replace H219/H224/H226/H231, reopen H212 automatically, or change a frozen candidate's formula.

## 1. Full-study loss with encoder-gradient replay

### Observed source, not a diagnosed performance defect

Read as text, without importing the trainer:

- `tools/l1_rco_stage0_v1.py:182` supplies `--train-windows 64`, `--train-sampling cover`, `--cover-stride 3`, `--eval-stride 3`, spatial attention, three slices and physical spacing 3.5 mm. The v2/v3 wrappers inherit this input recipe.
- `.local/scratch/l1-rco/k1v3-grouping/compact_train_l1.py:50` creates stride-based candidates per series/view, with training phase and neighbour jitter.
- The same file at lines 130–144 uniformly samples at most 64 candidates without replacement during training. Evaluation is not subject to that training cap.
- At line 445, class-specific attention uses a softmax across the included windows. At line 484, chunked encoder forwarding concatenates features before the head. Ordinary chunking retains the chunks' autograd graphs; it does not by itself make full-bag training memory-bounded.
- At line 1128 onward, the current objective is already **study-level** weighted BCE, followed by one backward, clipping, optimizer step, scheduler step and EMA update. It is not per-slice BCE. The proposed change is which windows jointly define that study loss.

Source SHA256 for the inspected trainer: `0d5fbae473eb1dc451871be0d610689d6baff8e6ac258acb091e66343ad0b791`. Four-file source audit: `.local/scratch/slot-a/gain-research-20261021/source-audit.json`. This is the same trainer identity reported in D-B-234, not evidence about a newly modified live run.

No completed candidate-count census, lesion-localization census or measured harmful-omission rate was obtained here. A capped bag can still contain adequate diagnostic evidence; random subbags can regularize usefully. We have not established that this mechanism limits K1, let alone the final ensemble.

### Why this is distinguishable from the earlier coverage experiments

D-B-121 measured omissions. D-B-125/134/156 removed omissions through coverage-preserving sampling, without establishing broad replicated improvement. D-B-234 correctly parked H212: merely porting a sampler to K1 is insufficient changed evidence.

The new comparison is instead:

`BCE(y, Head(all window embeddings))`

versus a weighted average of

`BCE(y, Head(each <=64-window subbag))`.

These objectives generally have different gradients, because both the attention normalization and the loss are nonlinear. Even guaranteed series coverage does not make them equivalent. A positive study label may be applied to a subbag lacking its focal evidence. Full-bag training removes the final 64-window cap, but still does **not** observe every voxel: the physical-stride candidate construction remains.

An illustration only: with 192 candidates and one diagnostic candidate, a uniform 64-window draw misses that candidate with probability 2/3. With three diagnostic candidates the probability is `C(189,64)/C(192,64) = 0.293965`. These are mathematical examples, not measured RSNA bag sizes or lesion counts. Windows can overlap, and a lesion can span many windows.

Primary support: Andersson et al., *End-to-end Multiple Instance Learning with Gradient Accumulation*, 2022: https://arxiv.org/abs/2203.03981 and https://arxiv.org/html/2203.03981v1 . The paper derives separable encoder gradients for an unchanged global attention objective. Its experiments use QMNIST and Imagenette bags, **not knee MRI**. It reports extra computation and warns about batch normalization. It supports feasibility, not a predicted medical AUC gain.

### Proposed replay boundary and critical implementation details

The following is a design sketch, not production code or a claim that the paper's implementation has been ported:

1. Fix the batch's studies, window IDs, padding/masks and augmentation parameters once. Stream each encoder microbatch under no-grad and retain its embeddings and reproducible random state.
2. The replay boundary is **after encoder plus `spatial_pool`**. For K1, this includes `sp_key`, `sp_pos`, `sp_q` and the moved `sp_head_norm`, not only the ConvNeXt trunk.
3. Treat the concatenated embeddings as differentiable leaves. Run the ordinary global head and the original target-normalized batch loss once; backpropagate into head parameters and retain `dL/dembedding`.
4. Replay each identical encoder/spatial microbatch with autograd and backpropagate its cached embedding gradient. Accumulate encoder/spatial parameter gradients. Do not recompute and accumulate the head parameter gradient once per chunk.
5. Clip once, then step optimizer/scheduler/EMA once. Restore the post-forward random state so replay does not perturb the next update's RNG sequence.

K1 has stochastic depth (`drop_path_rate=0.1`) and head dropout. Replay must preserve their exact stochastic realization, augmentation generator state, autocast, padding and train/eval behavior. Test the instantiated model for stateful/batch-dependent operations; source inspection is not runtime proof that all layers are suitable. Batch normalization cannot silently be frozen to obtain parity, since that would change the recipe. Existing internal gradient checkpointing must compose correctly with replay.

The current call moves the augmented full image batch to the GPU. Merely chunking the encoder would still allocate that tensor. A working implementation must also stream raw/normalized images and preserve the study-consistent augmentation, rather than repeatedly reading/augmenting different versions. CPU RAM, pinned memory and I/O costs need measurement. Embedding/head memory still grows with window count; this is not a claim of unlimited constant memory.

### Fixed discriminating experiment

The receiving session should independently review and register **one** candidate before building it. Keep target-file hash, patient split, initialization ancestry, optimizer, learning-rate schedule, study draws, batch size, crop, augmentation, EMA, updates, evaluation stride and serving formula fixed.

**Pair S versus F is the primary comparison:**

- For each study visit, generate the same complete stride-based candidate set and a seeded permutation. Split it into chunks of at most 64 windows. Both arms receive the same union of windows, once per update, and use the same replay machinery.
- **S (subbag control):** compute the existing attention head separately for each chunk. Combine per-target losses using fixed chunk-size/N weights, then apply the original study/target weighting across the batch. This is a matched-exposure control, not a claim to reproduce the original single-K64 training distribution.
- **F (full-bag candidate):** one head over the union and one study-level loss. Preserve the original target-normalized batch weighting.
- Use the same encoder microbatch shapes and random realization. Head/dropout differences caused by the changed bag objective are part of the intervention; do not claim bitwise equivalence between S and F.
- Both arms use one update per study batch. Do not compare one cheap 64-window run to a much longer full-bag run and attribute all differences to the objective. Reuse the existing K1 result only as contextual reference, with its different exposure disclosed.

This comparison controls window exposure and update count; head work differs and must be timed. If it is materially imbalanced, report it and add a predeclared compute-matched control only when needed to distinguish the cause. Do not introduce a post-result training-duration search.

Before training, a bounded **metadata-only** review may establish whether any relevant bags exceed 64, after the compute owner permits cache access. N<=64 everywhere would eliminate this cap-removal hypothesis. A high N>64 frequency alone is not a progression gate: D-B-234 already explains why incidence does not establish benefit.

**Numerical admission:** compare replay with normal full autograd on small bags that fit. Cover variable lengths/masks, target weights, an empty target column, padding, stochastic depth/dropout enabled, spatial parameters, one complete optimizer/scheduler/EMA step and resume RNG. First prove a deterministic toy case; then exercise the actual model/precision under the proper resource owner. Check all active parameter groups against the reference, including expected zero gradients. Pre-register numerical tolerances appropriate to FP64/FP32 and actual bf16 behavior; do not loosen them after looking at a failing result or disable randomness silently. A toy pass is not production qualification.

**Scientific progression:** two matched fits on one registered seed/fold are the initial cost. Complete the existing seven-lens readout, preserving exposed/ancestry limitations. A signal under the existing L1/L2/L3/L4 rules justifies the second matched seed pair, not submission. An F gain that does not survive the seed comparison, harms relevant targets, or disappears when inserted into the fixed final mixture is not a new finalist. Existing completed same-recipe seeds must all be reported. A predeclared per-target route is possible under L4; do not select routes using public scores.

**Cost:** two initial fits; up to two confirmation fits; validation/inference and one fixed final-mixture comparison. Encoder replay adds a forward pass relative to ordinary full autograd, and the number of windows per update can be larger. No hour estimate is defensible until loader-inclusive steps are measured. Current local HDD/remote throughput figures cannot be transferred blindly. No self-imposed training timeout is proposed.

**Serving:** inference architecture and windows can remain unchanged for a fixed serving stride. That is an advantage over adding another model, but actual branch execution, output validity and runtime still need qualification before any submission. This proposal is separate from ongoing submission readiness.

## 2. Knee-specific self-supervised pretraining

### What the new sources actually establish

**KneePreM**, submitted 2026-09-25, *Towards 3D Knee MRI Foundation Models via Large-Scale Unlabeled Pretraining and Label-Efficient Fine-Tuning*: https://arxiv.org/abs/2609.31461 . Its public abstract reports a 3D U-Net masked autoencoder pretrained on 19,011 unlabeled OAI series from 4,791 participants. Downstream sets include fastMRI+ (1,172 examinations), APM classification (1,716), and SKM-TEA/APM segmentation. It reports higher full-data macro ROC AUC than random initialization and SuPreM, with p<.001. The reported 0.722 fastMRI+ and 0.740 APM reduced-label results are different tasks and cannot be compared with RSNA 0.958.

**OAI-DINO**, ISMRM 2025 abstract 4014: https://archive.ismrm.org/2025/4014.html , DOI https://doi.org/10.58530/2025/4014 . Its public synopsis reports better classification transfer than a supervised-pretrained comparator and improvement over scratch. Full quantitative methods were access-limited; no unseen effect size is claimed.

The KneePreM abstract was inspected; its HTML paper was unavailable and no verified public code/checkpoint link was located in this search. Therefore **do not schedule this as a ready-to-download pretrained model** or claim an exact reproduction recipe. The OAI-DINO synopsis likewise does not establish a ready checkpoint. No weights or datasets were downloaded.

### Difference from our failed work

- D-B-244 tests 400 native-grade OAI adaptation updates followed by 800 RSNA recovery updates: mean delta about -0.00004 and arm rank correlation .999. It is negative for that fixed intervention. Earlier corrected OAI and auxiliary MRNet/KneeMRI routes also supplied adverse evidence (D-B-139, D-B-070/071/075).
- Those experiments supervise disease grades/labels. Masked MRI learning learns from withheld image content; it does not map MOAKS grades to RSNA targets and does not replace targets with teacher disease probabilities.
- D-B-180/187/191 already discourage generic teacher pseudo-label blending. H160 Mean Teacher is related consistency work, but predicting class outputs is not the same objective as reconstructing/predicting withheld MRI evidence.
- D-B-235 already proposed masked MRI-feature learning. The change here is stronger task-specific external support and a defined comparison, **not novelty of SSL itself**. Negative OAI transfer remains a reason for caution.

### Reviewable test, with its unresolved prerequisite explicit

First select and freeze one reproducible pretraining implementation and its exact architecture after reading its full method or available code. Prefer an image-only objective over text/diagnostic-token training. Do not mix a backbone change, a new cohort, a new label recipe and a new training schedule into one unexplained comparison.

For the chosen encoder architecture, compare:

1. **RSNA-only control:** common eligible initialization, followed by supervised RSNA training under the strong existing target recipe.
2. **MRI-SSL candidate:** the same initialization, a fixed masked-MRI pretraining stage, then the same supervised fine-tuning.
3. **Compute control:** an equal-compute supervised continuation with no external SSL, so improvement is not attributed to MRI SSL merely because it used more training. If initialized randomly, explicitly add the relevant strong pretrained control before claiming superiority to K1.

A faithful KneePreM reproduction uses a 3D U-Net family; a K1-compatible 2.5D adaptation is a different hypothesis and must be named as such. The paper does not justify projecting its improvement onto ConvNeXt or onto an already strong R7-initialized model. Choose one route, not a silent architecture sweep.

Use only an eligible participant-separated OAI/RSNA training pool; exclude every visit and both knees of evaluation participants. Training-only SSL is still exposure and invalidates an OAI evaluation panel containing those participants. Existing cohort/ROI/cache artifacts may be reused where their format matches the chosen method; do not assume that a 2.5D cache is a full native 3D volume. Any new files need their separate named source/size approval.

Mask a spatial region across adjacent slices/modalities, rather than leaving an almost identical unmasked neighbour that makes the task trivial. Preserve sequence identity and geometry. Check foreground reconstruction/feature prediction against a simple local-interpolation baseline on held-out training participants; a good reconstruction score alone is not a disease-classification gain. Fixed masking/task design and exposure accounting precede outcome inspection.

Run the initial comparison on one patient-separated fold, with fixed downstream updates and all twelve outputs. A pretraining loss decrease is only engineering progress. Continue to a second seed/fold only if the downstream seven-lens comparison and existing partner mixture provide a useful signal. Independent held-out image adjudication is particularly important when label-derived rulers disagree. Two-seed confirmation and actual serving qualification are required for finalist use.

**Cost and uncertainty:** at least one pretraining run, three downstream/control fits and their inference; more if confirmation succeeds. 3D activation memory, volume access, native preprocessing, sequence distribution shift and the exact compute match are unresolved. There is no justified GPU-hour or score forecast. If the implementation cannot be reproduced or the suitable volume pool is unavailable, retain it as a reserve rather than improvising a different objective mid-test.

## 3. Better labels: reuse the expert plan; do not confuse selection accuracy with truth

The useful existing control is the strong target recipe, including L2=75% Dread+25% Aquila where appropriate. D-B-095/098/111 found replicated improvement in several model families. Pure/high-Aquila changes were not consistently better (D-B-083/123/166). Gold oversampling and generic silent-label filling have adverse records (D-B-103/109/119/124/130). These controls must not be weakened to make a new source look good.

**New public lead screened:** Omer Zalman's `rsna-knee-11-label-evidence-fusion`, https://www.kaggle.com/datasets/omerzalman/rsna-knee-11-label-evidence-fusion . The dataset metadata reports 372/383 accepted cells correct out of 638 evaluated cells: 97.13% conditional accuracy at 60.03% coverage. It explicitly calls the 58-study panel familiar development data used during development. Synovitis is absent. Sources also include supporting research predictions, so ancestry requires investigation.

This is neither twelve-target independent label validation nor comparable to a full-population ROC AUC of .958. It is not evidence of fraud or uselessness; it is insufficient evidence to replace the current targets. Training only its accepted cells may change class/uncertainty selection and loss mass. A fresh blinded image panel and lineage-aware fixed comparison would be needed. Do not download/relabel/retrain just on the title's percentage.

**Recommended label route if resources become available:** reuse `docs/research/expert-mri-adjudication-plan-20261019.md` and the prepared H213 pilot96 protocol, preserving all recorded HOLDs. Two qualified readers inspect complete MRI, blinded to model/report outputs; adjudicate disagreements and retain uncertainty. Keep random evaluation separate from error-enriched training, including confident-agreement cases that expose shared mistakes. Replace selected training labels without duplicating Gold58 or increasing their sampling frequency. Compare to the strong current target recipe on the same eligible initialization, masks, updates and sampling; then test one fixed ensemble insertion.

D-B-235 already contains a 96-case feasibility design and conditional later study plan. Actual readings/new truth cells remain zero; the 200-case reserve is not magically clean, and teacher/encoder ancestry plus cross-Gold linkage remain unresolved. Removing the deadline objection does not remove these problems. This report does not dispatch readers or reopen selection. This is an existing high-potential reserve, not a fourth independent discovery of image relabeling.

## Other source leads and why they are not primary recommendations

| Lead | Verified observation | Consequence |
|---|---|---|
| 3DReasonKnee | Paper describes 7,970 OAI DESS volumes, approximately 494k region-question records, 100 manually annotated scans and nnU-Net-generated boxes for the larger set; MOAKS provides diagnostic grades | Useful possible **spatial-supervision** extension; neither 7,970 newly manually diagnosed RSNA cases nor independent text truth |
| KJD-25K | Official MICCAI 2026 page's dataset link points to `YiHui0124/KneeCoT` | Same prohibited lineage for this project; exclude, do not download under a new name |
| Med-SeAM | Full paper tables do not establish superior AUC over its strongest listed SSL comparator; central-slice selection discards anatomy relevant to this task | Do not promote headline accuracy improvements as evidence of twelve-target RSNA AUC gain |
| Generic latest forum tricks | Many repeat pseudo-labels, ensembles, crops and already audited code | No new qualified candidate inferred from anecdotes; D-B-256 remains valid within its deadline-focused scope |

3DReasonKnee primary sources: https://arxiv.org/abs/2510.20967 and https://arxiv.org/html/2510.20967v1 . The methods explicitly say experts manually annotated 100 scans, an nnU-Net generated the larger pseudo-label set, and boxes were derived from connected components. Queries/reasoning were developed from MOAKS criteria; they are not a new prospective adjudication of RSNA cases. A later experiment could supervise predicted anatomical regions while preserving a full-field branch, rather than only using OAI grades or a crop anchor. It must separate true manual masks from pseudo boxes, prevent target/case leakage, and compare against existing H216 anchoring and H163 geometry ideas. It is lower priority because actual eligible asset availability, domain transfer and incremental value are unresolved. No report text or dataset rows are needed for that review.

KJD source: https://papers.miccai.org/miccai-2026/0534-Paper0576.html . Med-SeAM source: https://dl.acm.org/doi/10.1145/3702250.3702276 . The source-screening conclusions concern the inspected versions; no source weights/code were executed.

## Evaluation contract for another session

Every proposal is a hypothesis. There is **no independent PASS** in this memo. Required builder/reviewer separation and project review/admission rules apply before important implementation/execution and before a result becomes decision evidence.

| Lens | Required treatment |
|---|---|
| L1 weak four | Existing fixed mean JEV/consensus rule; report per-target changes and uncertainty |
| L2 macro12 | Existing twelve-target rule on exactly aligned held-out rows |
| L3 alternate rulers | Report all existing eligible rulers; shared label/teacher ancestry is not independence |
| L4 target-specific | Existing >=.003 on >=3 of 4 rulers, without inventing a new mean gate; any selected route must be registered for confirmation |
| L5 seeds | Every existing same-recipe seed pair; if only one seed exists, mark confirmation missing rather than PASS |
| L6 bundle | One fixed incumbent/partner formula on the same rows; no weight grid, linear scaling of standalone gains, or public-LB tuning |
| L7 images | Gold58 is exposed/descriptive. OAI-trained/exposed arms cannot use overlapping OAI as independent evidence. Resolve ancestry or write the explicit ineligibility and use genuinely new image adjudication where needed |

Use `tools/multi_lens_readout_l4_v2.py` where applicable. Keep historical gates and the holistic ADVANCE/PER-TARGET/HOLD/DROP interpretation. A positive proxy is a reason for an appropriately bounded next test, not proof of public/private improvement. An unresolved material eligibility/runtime defect remains HOLD.

For full-bag replay, another session can start with this report, D-B-234/125/134/156, the four source files in the audit, and the applicable BUILDER-CHECKLIST.md/VERIFY-RULES.md. Review the discrimination between S and F first. If accepted, create a new versioned trainer and one concrete registered candidate; do not patch the submitted/frozen K1 trainer or a running job's bound files. No broad reread or repeat of settled coverage experiments is needed just to implement this named scope.

Research boundaries: no training, GPU diagnostics, model imports, submissions, score queries, new downloads, reader contacts or purchases were performed for this report. Existing B compute/queue ownership and paused scheduling were preserved. New plan items: **none — Taeyang requested proposals for review and execution in another session; the deliverable is D-A-052 and this report, not an automatically admitted experiment.**
