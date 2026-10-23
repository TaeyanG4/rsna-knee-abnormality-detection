# RSNA Knee Abnormality Detection — Plan

Keep only active unfinished plan items here — nothing else. Project-wide values such as the current best live in `handoff.md` Shared state. Every completed item leaves a discovery (positive, negative, or inconclusive) and a handoff event, then is removed from this file.

Each agent edits only its own `## Agent: <name>` section. Plan item IDs use `H-<Agent>-<NN>`, for example `H-A-01` or `H-B-07`.

Before adding an item, confirm it is not already in `discoveries.md` (every completed experiment is there, including negative and inconclusive ones) or already queued under another agent (check headings and `Hypothesis` lines only). A repeat needs a real `Improvement` and a citation of the earlier ID.

Priority formula:

```text
Priority = 2*Impact + 2*Information + Confidence + Unblock + Diversity + (3-Cost)
```

All factors are 0-3. Cost is 0 for cheap and 3 for expensive.

Use this exact item format:

```markdown
### H-A-01 — Short title
- Sources: D-B-001, D-A-004 / none
- Hypothesis: ...
- Evidence: ...
- Improvement: ...
- Impact: 0-3
- Information: 0-3
- Confidence: 0-3
- Unblock: 0-3
- Diversity: 0-3
- Cost: 0-3
- Priority: ...
- Resource: CPU | GPU | EITHER
- Parallel: YES | NO
- Other: NONE | <external executor>
- Next test: ...
```

Example external executor: `Other: Kaggle`.

## Agent: A

## Agent: B

### H-B-27 — Alternative report-label set: CoAtNet B8 5-fold trained on the flight hybrid labels instead of Dread, judged on public
- Sources: D-B-013, D-B-014, D-B-027, D-A-049
- Hypothesis: Training the unchanged CoAtNet B8 5-fold recipe on the flight hybrid report labels (gold rows removed) instead of the Dread labels changes the public score by ≥ 0.009 relative to CoAtNet5 on Dread (0.914).
- Evidence: every own model so far was trained on Dread labels or Dread-derived mixes (EXP061/EXP072/EXP084/EXP099 summaries), so a label-set change has never been tested; the forum reports +0.015 public from label work combined with an external public dataset, not a pure label-set swap; the earlier "+0.011" is not in any retained snapshot and is withdrawn (updated 21:42 KST, 21:05 sync / D-B-013); Dread differs strongly in prevalence from the other report-derived sets (effusion 0.33 vs 0.58, Baker's 0.17 vs 0.36, OA lower; D-B-014) and agrees with binarised flight only at macro AUC 0.933; non-label levers tested today all stayed inside ±0.004 public (D-B-027) and local OOF cannot score a label change (D-A-049), so public is the right instrument here.
- Improvement: first single-variable label-set change in our pipeline (same recipe, folds, seed, K36 serving; pos_weight follows the label set's own prevalence because it is part of the label definition); one pre-registered submission paired with an already-scored control (56893801).
- Impact: 3
- Information: 3
- Confidence: 1
- Unblock: 1
- Diversity: 2
- Cost: 2
- Priority: 17
- Resource: GPU
- Parallel: NO
- Other: NONE
- Next test: running: label set flight_v4 registered (storage/labels/derived_EXP002/flight_public_control_weak4349.csv = raw flight v4hybrid minus the 58 gold rows, 4,349 rows, gold overlap 0, pos_weight from the labels), folds 0–4 queued as artifacts/experiments/LOCAL_GPU_CHAIN_V11.json (≈3.75 h); then assemble, release v5 at K36, Public3, ONE submission. PRE-REGISTERED (2026-10-07 12:20, before any result): adopt flight labels for later own models if public(flight5) − 0.914 ≥ +0.009; reject if ≤ −0.005; otherwise inconclusive and Dread stays (both may be ensembled only on OOF-free grounds later). The flight-label OOF is scored against flight labels and is not comparable with the Dread OOF, so no OOF-based decision is taken. ORDER (changed 2026-10-07 13:10 KST, direction review 13:05): folds 2–4 run after H-B-28's relabel 5-fold (flight F0 done 13:08, F1 running); its submission moves to after the 09:00 KST reset. INTERPRETATION (16:49 KST, same source as H-B-28): flight_v4 also has label-derived class weights (Effusion 2.07→1.00, PF OA 1.80→1.06, Baker's 4.87→1.79), so its public result measures labels + weights. JOINT READING (17:15 KST; source: 17:05 direction review, fable): read relabel_v1 (tonight), flight_v4 (after the reset) and Dread CoAtNet5 0.914 together as one three-arm label test; if all three lie within ±0.005 the label branch closes and H-B-29 is the only label-set product. GPU ORDER kept (17:15): flight F2–F4 run after EXP110 (fable: two folds already spent, completes H-B-27/H-B-29; astra preferred starting the RadR50 line first — not adopted because RadR50's standalone public value arrives with H-B-31 tomorrow and decides that investment).

### H-B-29 — Label-set ensemble: rank-mean of the three CoAtNet 5-folds trained on Dread, flight and relabel_v1
- Sources: D-A-022, D-B-021, D-B-014, D-B-016, D-B-027
- Hypothesis: An equal-weight per-target rank mean of the CoAtNet5 models trained on Dread, flight_v4 and relabel_v1 labels scores at least as well on public as the best single label set, and is more robust on private than picking one set by one noisy public score.
- Evidence: one public score cannot separate label sets reliably (weak-label proxy band ±0.007–0.010, D-B-016; adoption bar +0.009); decorrelated members transfer their rank-mean gain to public (E1 +0.004, D-B-021); report labels and image-informed signals are complementary on gold (+0.029, D-A-022); the three label sets disagree strongly (53% of studies flip ≥1 decision between Dread and relabel_v1, positive rates differ 2–3× on OA/effusion), so their models should be maximally decorrelated; the host warns that prevalence differs between public and private, which favours an ensemble over a single-set pick. Proposed by fable in the 2026-10-07 13:05 direction review.
- Improvement: no new training; replaces "adopt one label set on one public score" with an ensemble whose decision rule is fixed before any of the three label-set models is scored.
- Impact: 2
- Information: 2
- Confidence: 2
- Unblock: 1
- Diversity: 2
- Cost: 1
- Priority: 15
- Resource: CPU
- Parallel: YES
- Other: Kaggle (release Public3 + one submission)
- Next test: after H-B-27 and H-B-28 are assembled: K36 OOF rank mean on the 4,343 studies, reported per target against each of the three label sets (breakage check, no selection); release with three CoAtNet5 arms (runtime ≤ the v5 budget measured on Public3); ONE submission. Pre-registered reading: compared with the best single-label-set CoAtNet5 public score; weights fixed equal and never tuned on public. (Changed 17:15 KST after astra's 17:05 review found conflicting promotion rules between H-B-29 and H-B-32: no automatic default — the own stack for the final blend is chosen once in H-B-35 by pre-registered evidence.) DIAGNOSTIC NOTE (18:43 KST; source: answer to the user's question on validating label changes): the 5-fold ensembles never saw the Gold58 studies (excluded from every label file), so a paired Gold58 comparison of the label-set models may be reported as a direction check, never as a gate (H-B-26 rule; width ±0.05). RUNTIME NOTE (20:13 KST): three separate CoAtNet5 arms do not fit the 9 h screen (per arm 1.3 × (6.525 + 5 × 0.35) × 1300 + 600 ≈ 13,985 s → three arms ≈ 41,955 s > 30,600 s; the 6.5 s/study per-arm decode dominates), so the release needs either one shared decode for all 15 members with per-member probabilities rank-combined per label set (an EXP105/v6-style worker change) or faster preparation (H-B-41); a single 15-member mean_prob arm would weight label sets by their probability scales and is NOT the pre-registered equal rank mean. STATUS (22:04 KST): relabel5 public 0.913 (D-B-038, inconclusive vs 0.914); flight5 Public3 PASS (v3, 22:01), submitted after the 2026-10-08 reset; the three-arm reading follows its score. RUNTIME UNBLOCKED (02:56 KST, D-B-048): v7 kernel .local/exp119-release-v7/coat5fl5rl5 projects 20,500 s (< 30,600 s); decision still waits for the flight5 public score (09:02 submission) per the registered joint reading.

### H-B-31 — Release and submit CoAtNet5 + RadR50x5 (equal rank mean, K36)
- Sources: D-B-030, D-B-027, D-B-016, D-B-020
- Hypothesis: The equal-weight rank mean of CoAtNet5 and RadImageNet ResNet50x5 (both Dread labels, K36) scores at least +0.006 above CoAtNet5 alone (0.914) on public.
- Evidence: OOF on 4,343 studies 0.8990 vs 0.8875, +0.0115 [+0.0096, +0.0134] (D-B-030), larger than the weak-label proxy band (D-B-016; conditional on fixed predictions); RadR50 alone 0.8914 is our highest OOF family (descriptive: platform and schedule differ, D-B-008); the per-target split (RadR50 MCL/lateral meniscus/OA, CoAtNet ACL/effusion/Baker's) survives the component bootstrap within the Dread contract (9 of 12 targets); caution: the ResNet34 OOF gain +0.003 did not transfer (0.914 both, D-B-027).
- Improvement: first release of a RadImageNet member; the largest own-model OOF gain so far; decided by one pre-registered public score.
- Impact: 3
- Information: 2
- Confidence: 2
- Unblock: 2
- Diversity: 1
- Cost: 2
- Priority: 16
- Resource: CPU
- Parallel: YES
- Other: Kaggle (release Public3 + one submission)
- Next test: build release v6 = v5 + a RadR50 arm (centre cache plane ×3, 2x−1 normalisation, ResNet50 MIL, K36, mean_prob over 5 folds, runtime screen per arm) by a sub-agent with tests and an independent review; T4 serving parity on stored OOF studies; Public3; ONE submission after the 09:00 KST reset. Pre-registered (16:17 KST): transfer confirmed if public ≥ 0.920; no transfer if ≤ 0.914; in between = inconclusive. If confirmed, CoAtNet5 + RadR50 becomes the own-model arm for the anchor blend (H-B-32); any use in H-B-29 or a final selection goes through H-B-35 (H-B-29 promotion removed 21:42 KST, C issue 16:55/17:24). ADDED (16:30 KST; source: the user asked whether single models reach 0.95 per the forum and what is most likely to get there; reason: the forum reports RadImageNet + centre slice ×3 as a strong input recipe (+0.020 vs adjacent slices, +0.063 vs ImageNet R50; D-B-013 — one participant's fold-0 OOF, contradicted by another participant, (updated 21:42 KST, 21:05 sync / D-B-013)) and a ResNet50 224 5-fold at 0.954 public, while our RadR50x5 is our best OOF family (0.8914, D-B-030) — whether this family alone is near the forum level is the single most direction-setting unknown; evidence: D-B-013, D-B-030): the same release also serves RadR50x5 ALONE as a second kernel and gets ONE submission. Pre-registered: RadR50 alone ≥ 0.930 → the family is near the forum recipe and becomes the main own-model line (seed/teacher/label work moves to it); 0.914–0.930 → useful ensemble member only; ≤ 0.914 → no better than CoAtNet5 on public. RESULT (2026-10-08 10:48 KST, D-B-049): rad5 public 0.925 (ensemble member band); coat5rad5 pending (two accidental copies, D-B-055). RESULT 2 (2026-10-08 12:41 KST, D-B-057): coat5rad5 public 0.926 (≥ 0.920 transfer confirmed; only +0.001 over rad5). Both H-B-31 readings done; registry bound. Item complete except bookkeeping.

### H-B-32 — Anchor + CoAtNet5/RadR50 rank blend with a hard four-child gate (conditional on H-B-31)
- Sources: D-B-031, D-B-030, D-B-018, D-A-048
- Hypothesis: Blending the unchanged anchor with the stronger own pair CoAtNet5 + RadR50x5 (instead of CoAtNet5 + ResNet34x5) at a pre-registered rank weight beats the anchor alone on public.
- Evidence: the 0.914-level own arm scored 0.001 lower at weight 0.2 (0.942 vs 0.943, one run, D-B-031 updated 21:42 KST; a mild caution against anchor blends with weak own arms); CoAtNet5 + RadR50 OOF 0.8990 vs 0.8905 for the arm used in D-B-031 (D-B-030); RadImageNet overlaps one anchor phase (D-B-005), so its marginal value inside the anchor is uncertain; shared decoding keeps the run near 8 h (D-B-018).
- Improvement: stronger and more complementary own arm; the four anchor CoAt children become a hard gate (D-A-048 gap); only after H-B-31 shows the own pair transfers to public.
- Impact: 3
- Information: 2
- Confidence: 1
- Unblock: 1
- Diversity: 1
- Cost: 2
- Priority: 6
- Resource: CPU
- Parallel: YES
- Other: Kaggle (Public3 + one submission)
- Next test: only if an own stack has standalone public ≥ 0.920 with OOF support (H-B-31 CoAtNet5 + RadR50 is the current candidate; changed 17:15: the own arm is whichever stack H-B-35's rule selects): extend EXP105 with a RadR50 arm on the shared decode, make the four-child check fail the blend (anchor-only fallback otherwise), re-measure runtime at 100 studies, then ONE submission at the pre-registered weight 0.7 anchor / 0.3 own (fixed now, 17:06 KST, not tuned on public); reading: ≥ 0.946 (+0.003 over the anchor) = blend kept as a final candidate, ≤ 0.943 = drop anchor blends. THRESHOLD (2026-10-08 01:10 KST, before any H-B-31 score): kept at standalone ≥ 0.920 as registered — fable proposed letting 0.917–0.919 trigger; B declined because the 0.2-weight blend with a 0.914 arm already scored below the anchor (D-B-031) and astra advises against weak arms for B. SHARED DECODE (2026-10-08 01:38 KST, both 01:05 reviewers: runtime is the critical path): stage 1 started — release v7 (tools/exp119_release_v7.py, CPU sub-agent): one DICOM decode per study feeding any number of CoAtNet/RadR50 arms, parity vs v5/v6 separate workers, runtime projections for 2–4 arms; stage 2 (anchor + own arms in one kernel, from EXP105) after stage 1 and the 2026-10-08 scores. DIVERSITY NOTE (2026-10-08 01:47 KST): prefer own arms that differ from the anchor families (Raptor CoAtNet v5/v8/v10, DINOv2 group, frozen-RadImageNet group): RadR50 (fine-tuned, centre slice) is the best available but overlaps the anchor's RadImageNet encoder; a second RadR50 seed reduces variance, not correlation. STAGE 1 DONE (02:56 KST, D-B-048): release v7 shared decode bit-identical to v5/v6, projections fit; stage 2 needs the store-reading worker inside EXP105 + a 100-study 2×T4 timing (screen constants put coat5+rad5 at ≈49 min/GPU inside the anchor). ARM DECISION (2026-10-08 11:49 KST, registered before the coat5rad5 public score; decision consult fable + astra): blend #1 = rad5-only own arm (RadR50x5, plane-aware flip TTA on only if the measured 2×T4 timing keeps the kernel ≤ 30,600 s), 0.7 rank(anchor) + 0.3 rank(own), ONE submission after Public3 PASS and the screen; blend #2 = coat5rad5 arm, ONE submission, only if coat5rad5 standalone public ≥ 0.920 (registered gate) AND a measured full-scale timing with that arm ≤ 30,600 s (projection 31,470 s fails today) AND receipt/screen pass; a disappointing blend #1 does not cancel blend #2 when its conditions hold; between qualifying blends prefer rad5-only unless the coat5rad5 blend is ≥ +0.003 higher on public; selection B still needs ≥ 0.946; selection A stays the unchanged anchor 56773226. coat5rad5 SCORED (2026-10-08 12:41 KST): 0.926 ≥ 0.920 but only +0.001 over rad5 (D-B-057); blend #2 remains conditional on the timing condition (projection 31,470 s > 30,600 s) — blend #1 rad5-only proceeds once EXP126 passes its tests, parity and a runtime decision. DEFERRED (2026-10-08 12:48 KST, same consult): no Kaggle run or slot is spent on the rad5-only blend now (its +0.001–0.003 is unreadable at public noise, runtime margin 54.7 min, and it would use most of this week's Kaggle GPU quota); EXP126 engineering is finished on CPU (anchor-cell identity, hard four-child gate, runtime guard, Public3 plan); ONE blend with the best bundled own arm by 2026-10-16 (ERROR rerun possible before 10-18); if the 10-12 milestone fails, blend rad5 + TTA + multi-head on 10-13. Supersedes the 11:49 'blend #1 proceeds' timing (arm choice rules unchanged). ON HOLD (2026-10-08 13:22 KST, user direction): the anchor is no longer used as a BLEND partner — our anchor blends never gained (0.942), the anchor costs ≈7.3 h of the 9 h budget and overlaps the CoAtNet/Raptor/RadImageNet families; it stays only as final selection A (already scored, zero cost). EXP126 work is parked; reopen only on the user's request.

### H-B-33 — Prospective public-Raptor recipe bundle (public training UIDs/labels unknown, D-A-050): CoAtNet 224 on all 4,343 studies, 16 epochs, SWA of 1-based epochs 13/15/16, K36 (H-B-30 did not pass, D-B-032)
- Sources: D-A-050, D-B-019, D-B-017, D-B-027, D-B-013
- Hypothesis: Training our CoAtNet-2 224 recipe the way the public v5 model was trained — all labelled G1 studies, ≥15 epochs, an average of three late checkpoints (12, 14, 15; fixed in advance, not chosen on Gold58), served with all 62 positions if H-B-30 passes — raises a single model to ≥ 0.923 public, i.e. the 0.914 → 0.928 same-label gap is a recipe bundle.
- Evidence: the public 64-slice Raptor scores 0.928 on our labels and cache contract (D-B-019) and is an SWA of epochs [14,12,15] served at K62 across slots (D-A-050, verified), while our runs train on ≈80% per model for 8–12 epochs with a fixed final checkpoint. DUPLICATE CHECK (17:15 KST, user rule): the components were already tried one at a time and each stayed inside the 0.006 seed swing (D-B-017) — SWA on EXP060 fold 0 0.8963 vs best checkpoint 0.8960 (D-A-017 update), 16 epochs −0.005 vs B (D-B-026), K36 +0.0034 (D-B-012), 80 slices −0.003 (D-B-023); data fraction is queued as local one-fold contrasts H-A-18 (A) and H-C-05 (C, 64% vs 80%). Untested: training on 100% of the studies and the factors combined, judged on public. Proposed by fable in the 17:05 review.
- Improvement: the only parts not covered elsewhere are 100% training data and the combined factors on public (H-A-18/H-C-05 test 64% vs 80% locally on one fold); per-epoch checkpoints also enable later SWA tests and Colab resume.
- Impact: 2
- Information: 2
- Confidence: 1
- Unblock: 1
- Diversity: 1
- Cost: 2
- Priority: 12
- Resource: GPU
- Parallel: NO
- Other: Kaggle (Public3 + one submission)
- Next test: build (CPU sub-agent tonight) a runner/protocol: B8 CoAtNet-2 224 recipe, Dread labels and Dread pos_weight, all 4,343 G1 studies (Gold excluded, no validation, no Gold or OOF checkpoint selection), 15 epochs with per-epoch checkpoints, SWA = mean of the epoch-12, -14 and -15 weights with BatchNorm statistics recomputed on the training data; local GPU after flight F4 (≈1.8 h); release (v5 arm, K36 (H-B-30 did not pass, D-B-032), else K36) → Public3 → ONE submission. Pre-registered (17:15 KST): ≥ 0.923 = the recipe bundle reproduces most of the gap → the 5-fold/ensemble line moves to this recipe; ≤ 0.914 = the gap is not this bundle; in between = inconclusive. CHANGED (18:08 KST; source: independent review of the EXP112 trainer, MAJOR M1; reason: the public v5 swa_over [14,12,15] is zero-based (public trainer code.py.txt:509,622; HA14 audit.md:66), i.e. 1-based epochs 13/15/16 of a ≥16-epoch run, so 15 epochs with SWA 12/14/15 would not reproduce the target; evidence: D-A-050 and the review): run EXP112 V2 = 16 epochs (OneCycle over 16) with SWA of 1-based epochs 13, 15, 16 (tools/exp112_full_swa_v2.py, training/runners/exp112_full_swa_v2.py, protocol EXP112_FULL_SWA_PROTOCOL_V2, tests 148 pass; V1 kept on disk unused), ≈2.0 h, queued in LOCAL_GPU_CHAIN_V15 after flight F4. FOLLOW-UP CANDIDATE — WITHDRAWN 21:54 KST by both 21:05 reviewers (Gold58 selection bias D-A-001, ±0.05 width D-A-017, costs a submission); original text kept for provenance: (18:43 KST; source: answer to the user's question on how others reach 0.95; evidence: D-A-050 — the public v5 chose its three SWA epochs by Gold58 AUC): if H-B-33 is inconclusive or positive, test choosing the 3 of 16 per-epoch checkpoints by Gold58 macro AUC (a low-dimensional choice, 560 subsets) against the fixed 13/15/16, judged on public once; Gold58 exposure and its ±0.05 width (D-A-017) make this a deliberate exception to H-B-26's no-Gold rule that needs its own pre-registration before use.

### H-B-34 — RadR50 line on the local GPU (conditional on H-B-31's standalone score)
- Sources: D-B-030, D-B-013, D-B-008
- Hypothesis: The RadImageNet centre-slice family improves further with the single-model levers (window policy, SWA/epochs, leak-free distillation with the teacher-scale issue fixed), each measured against a local control.
- Evidence: RadR50x5 is our highest OOF family (0.8914, descriptive) and one participant's fold-0 OOF result, contradicted by another participant (D-B-013, (updated 21:42 KST, 21:05 sync / D-B-013)); Kaggle-trained folds cannot be a matched control for local runs (D-B-008), so a local control 5-fold comes first (astra 17:05); fable advised waiting for the standalone public score before spending ≈3.5 h on it.
- Improvement: moves the strongest family to the lane with free compute and adds a matched control.
- Impact: 3
- Information: 2
- Confidence: 2
- Unblock: 1
- Diversity: 1
- Cost: 2
- Priority: 16
- Resource: GPU
- Parallel: NO
- Other: NONE
- Next test: CPU tonight: port tools/exp107_kfold_radr50.py to the local queue (protocol copy with the local device, same seed/folds/labels). Tooling ready (17:39): tools/exp113_radr50_local.py + training/runners/exp113_radr50_local.py, protocol EXP113_RADR50_LOCAL_PROTOCOL_V1 (EXP107 copy, exact seed-42 init enforced, 45 tests pass, CPU smoke 2 updates), ≈20 min per fold, ≈1.7 h for five. START RULE (changed 17:39 KST; source: compute-lane rule — after flight F4 and H-B-33 the GPU would sit idle overnight until H-B-31's score arrives ≈13:00 tomorrow; reason: at ≈1.7 h the local control costs only otherwise-idle GPU time; evidence: the builder's timing estimate): queue the local control 5-fold right after H-B-33; H-B-31's RadR50-alone score then decides only the follow-up variants (≥ 0.914 → windows/SWA/distillation on this family; ≥ 0.930 → main line ahead of H-B-33 follow-ups). FAST-PLATFORM DESIGN (19:38 KST; source: the user asked whether short, fast models could carry preprocessing/label work and whether Colab helps; reason: at ≈20 min per fold RadR50 allows 5-fold × 2-seed comparisons on all 4,343 studies, which fold-0 single-seed CoAtNet contrasts never had (seed swing 0.006, D-B-017); Colab rejected for now — sessions were reclaimed after 30–40 min, a T4 fold takes ≈75 min, the 32 GB cache must be re-staged each session, and Colab arms cannot pair with local arms (D-B-008)): after the local control, every RadR50 variant runs 5 folds × seeds 42 and 43 against the same two control seeds; non-label changes are decided on pooled OOF (Dread), label changes are not decidable locally and still need public. DUPLICATE CHECK: no own crop-size comparison exists (the cache is fixed at 140 mm, forum 100 → 140 mm helped, D-B-013); no RadR50 second seed; distillation was only tested on CoAtNet (D-B-011, D-B-027 with the teacher-scale flaw of D-A-047), and slot A H-A-15 designs the leak-free scale-separated version — a RadR50 student would cite and reuse it and be scored against the original Dread labels. Order (updated 21:06 KST from D-B-036; DUPLICATE CHECK: the only augmentation test, D-B-009 geometric on Kaggle T4, was confounded by device/stack, so augmentation is untested locally): control seed 43 → augmentation: port only changes confirmed in H-B-48 (Colab, 22:56 KST) — not screened again locally; formerly: augmentation stack ablations one change at a time (generic and MRI-specific) as matched short screens (fold 0, same seed and schedule as the control fold) → second-seed confirmation of any screen gain → full 5-fold only for confirmed variants (fable + astra 21:05 review, adopted 21:54 KST, astra) → crop size → epochs/SWA → leak-free distillation (now the targeted form H-B-46, 22:34 KST) → RadR50 × relabel_v1 (local again, 21:37 KST, H-B-39 on hold after the Colab session loss). MULTI-REFERENCE RULE (19:43 KST; source: the user noted that Dread labels are not guaranteed correct; reason: systematic Dread errors can reward a model that copies Dread rather than the official read (the earlier "plausible cause of the CoAtNet5 + ResNet34x5 miss" rationale is withdrawn 21:42 KST: that candidate passed the multi-reference line and still did not transfer, D-B-033; the rule stays as a sanity guard, H-B-42 carries the transfer question); evidence: report-derived label sets reach only ≈0.88–0.90 macro AUC against the 58 official labels and disagree with each other (D-B-014), and three independent references now exist): a non-label change is adopted only if pooled OOF improves against rint(Dread) with the component-bootstrap CI above 0 AND does not fall by more than 0.002 against flight_v4 and relabel_v1 (each binarised at 0.5); all three share the report-text ceiling, so report-vs-image gaps remain visible only on Gold58 (direction check) and public. DUPLICATE CHECK: D-C-001 recommends fixed evaluation references; H-B-26 scores label candidates against their own set — this rule extends both to every non-label decision. GPU ORDER (21:42 KST, recorded at the 21:05 sync): H-C-05 (slot C, priority 17) seeds 42/43 run on the shared local GPU before EXP113 F0 (this item, 16), chain V16 (handoff 21:37 event). MAIN LINE (22:42 KST, user direction): after the H-B-42 ruler is validated, this item is the single-model improvement line, run by the rank-12 protocol (D-B-036): one simple fast CNN (RadR50) → seeded control (seed 42 + 43) to set the ruler resolution → one change at a time as fold-0 matched screens on two seeds → five folds only for confirmed gains → submit only real improvements, bundling small ones; order: augmentation (generic, then MRI-specific) → crop/resolution → epochs/SWA → label handling (H-B-46/H-B-47) → backbone last. Judging uses several rulers together (the best H-B-42 ruler must improve; none of the others may drop by more than its resolution). COLAB LANE (22:51 KST, user direction): the single-model improvement loop is run by the user in Colab, decided together in chat, on the compact 12-slice ResNet34 pipeline (docs/learning/colab_knee_ablation_v1.1.0.ipynb: fold-0 matched screens, seeds 42/43, four rulers). To avoid duplicate work the local GPU does not repeat those augmentation screens; it ports only Colab-confirmed changes to RadR50 (transfer test) and runs the other plan items. RULER (23:07 KST, D-B-042): adopt a RadR50 change if pooled 5-fold Dread Δ ≥ +0.004 vs the matched control with bootstrap CI > 0 and consensus Δ ≥ 0; expected public ≈ 0.914 + Δ ± 0.010. REVIEW 01:05 (both reviewers): after EXP113 → RadR50 seed-43 local 5-fold (EXP118, tooling being built by a CPU sub-agent; queued as soon as ready, ahead of the VLM pilot) → H-B-47 F0; GPU chain V19 (01:09): EXP113 F3–F4 → EXP117_MULTIHEAD_F0 → EXP115 VLM pilot (bounded). USER DIRECTION (2026-10-08 11:52 KST): focus on raising the own single model; first screens EXP127 (astra building): ADJ3 adjacent-slice input (forum +0.018 OOF for centre-adjacent over spread slices), EP40 (forum 25→50 epochs +0.004), AUG1 (affine + gamma + plane-aware flip); fold 0, seeds 42/43 vs matched local controls EXP113 F0 / EXP118 F0; Tier A/B verdicts; winners combined → one five-fold → own-arm release; GPU order vs the VLM pilot under consult (fable + astra). GPU ORDER (2026-10-08 11:54 KST, consult fable + astra, both option B): after the committed jobs, EXP127 screens AUG1 first (augmentation never tested on RadR50; plane-aware flip already +0.0021), then ADJ3 (contrary priors: D-B-028 macro-flat on CoAtNet, a participant reported centre×3 beating adjacent), then EP40 (the forum's 25→50-epoch +0.004 was withdrawn in D-B-013 — low prior, doubles downstream cost); freeze a ≤ 3-change bundle → one five-fold vs matched controls → the exclusive capped VLM pilot overnight (no concurrency, it would falsify its speed gate). Fallback: if EXP127 is not validated when EXP122 s43 ends (≈15:25), the VLM starts then and the screens follow (≈19:30). SINGLE-MODEL LINE DECISION (2026-10-08 12:48 KST; user proposal 'single model ≥ 0.94, then no anchor'; consult fable + astra + B all choose (b)): main line = own RadR50 single model; realistic 10-day range ≈0.930–0.935, ≥ 0.94 a stretch (fable ≈20–25%). MILESTONE 2026-10-12 KST: a jointly evaluated five-fold bundle (≤ 3 changes, two seeds for training changes) vs frozen rad5 controls: ΔDread ≥ +0.008 (forecast ≥ 0.933) = on track; +0.005…+0.008 = continue; < +0.005 = stop budgeting around 0.94 and reopen the blend path at once. One public reading of the bundle by 2026-10-14 with a pre-registered forecast 0.925 + ΔDread; if it lands > 0.005 below the forecast, stop the accumulation line. Rule-compatible levers added from D-B-058: all-data + late SWA (swa16 reading tomorrow), JEV-blended labels if publicly available (Tier A), dense slices/more windows, mirror TTA (have). USER DIRECTION (2026-10-08 13:22 KST): 'we have a reasonably accurate validation ruler — keep improving through it': the RadR50 single model is THE main line; every change is judged on the local ruler (pooled five-fold Dread + consensus/flight/relabel, Tier A/B rules, joint bundle runs); public submissions only for bundle readings and registered forecasts; continuous loop: each finished screen → verdict → next screen queued immediately (EXP127 AUG1 → ADJ3 → EP40, then K sweep H-B-56, multi-head seed check H-B-47, label head H-B-55 judged on consensus + public). AUG1 ADOPTED (2026-10-08 19:07 KST, D-B-066): EXP134 five-fold two-seed ΔDread +0.0087 [+0.0068, +0.0107], Tier A met. Ruler-based forecast for an AUG1 RadR50 2-seed×5 + plane-aware flip TTA release (registered now, before any submission): public ≈ 0.925 (rad5) + 0.004 (two seeds, D-B-060) + 0.009 (AUG1) + ≈0.002 (TTA, D-B-056) ≈ 0.938–0.941; a reading ≤ 0.932 would mean the AUG1 gain did not transfer. Next: release kernel + private dataset of the ten AUG1 checkpoints (sol), one public reading 10-09 (non-OAI → candidate for selection A and the third within-family ruler test); bundle EXP138 running on top. EVIDENCE-PANEL RULE for label-mix changes (registered 2026-10-09 01:26 KST, BEFORE the EXP144 Gold58 and JEV-reference results): the Dread ruler is home-biased when one arm trains on Dread only, so a label-mix change (first case: EXP138 bundle vs EXP134 AUG1 single head) is judged on a PANEL, no single source decisive: E1 Dread OOF — tolerance, must not be worse than −0.003; E2 independent LLM label sets used by NEITHER arm — flight_v4 (+0.0113 [+0.0098, +0.0125], already known) and JEV HYBRID binary (computed after this registration) — both positive with lower 95 % bound > 0; E3 Gold58 official labels from fresh inference (EXP144; n=58, direction only) — point estimate not below −0.005; E4 external expert labels (H-B-65: OAI MOAKS / fastMRI+ / MRNet ACL) when available — not worse; E5 partner proxy and circular references (consensus, relabel_v1) — report only. ADOPT the candidate as the RadR50 line when E2 passes and E1/E3 (and E4 if available) stay within tolerance; otherwise keep the incumbent. Public reading stays the final arbiter for final candidates (submission-value rule). To be ratified by fable + astra at the 05:00 review.

### H-B-35 — Final two selections (rule fixed now)
- Sources: D-B-031, D-B-025, D-B-016, D-A-049
- Hypothesis: A selection rule fixed before the last week protects the private score better than picking the best public variants.
- Evidence: Kaggle keeps the better private score of the two selections; one public score moves within a weak-label proxy band of ±0.007–0.010 (D-B-016); the anchor's own rerun variation is unmeasured (D-B-031 updated 21:42 KST); unexplained hidden-rerun errors happen (D-B-025); both reviewers (13:05, 17:05) recommended an unchanged-anchor safe pick plus one higher-variance own-stack bet.
- Improvement: replaces ad-hoc promotion rules in H-B-29/H-B-32 with one rule.
- Impact: 3
- Information: 1
- Confidence: 2
- Unblock: 2
- Diversity: 0
- Cost: 0
- Priority: 16
- Resource: CPU
- Parallel: YES
- Other: NONE
- Next test: rule (17:15 KST, from fable + astra 17:05): A = the unchanged anchor 56773226 (COMPLETE 0.943); B = the anchor blended at a fixed 0.3 rank weight with the own stack that has the best pre-registered evidence (pooled OOF on 4,343 studies AND a standalone public score ≥ 0.920), never simply the best public among variants; if no own stack reaches 0.920 standalone, B = the best COMPLETE own-stack submission or a second anchor variant chosen by the same evidence; both selections COMPLETE by 2026-10-20 and frozen ≥ 48 h before the deadline. SCREEN (19:57 KST, from D-B-033; amended 21:42 KST after the 21:05 sync): every candidate runs tools/submission_screen.py before its submission; since 23:13 KST the four gates fail closed (D-B-033 update), so SUBMIT-ELIGIBLE or PUBLIC-ONLY (no-OOF models with a declared reason and checkpoint sha) is required and HOLD/BLOCK stops the submission; registered_at must carry a time zone and precede the run; the screen report is linked in the submission's handoff event. EFFICIENCY PRIZE NOTE (20:05 KST, D-B-034): a separate efficiency prize evaluates one of the two final selections by AUC and runtime; with own models at 0.91x it is out of reach, so the rule above stays accuracy-first; revisit only if an own stack under ≈1 h reaches ≥ 0.945 public. AMENDED (fable + astra 21:05 review, adopted 21:54 KST): B is no longer automatically the 0.3 anchor blend — B is the strongest complementary COMPLETE candidate with credible validation (pooled OOF + consensus support + runtime margin); the anchor + own blend at the pre-registered 0.3 weight qualifies only with own standalone ≥ 0.920 public plus OOF and consensus support; never a 0.91x submission merely for diversity; weights never tuned on public; submit B by 2026-10-18 so an ERROR rerun can be repeated before the 10-20 COMPLETE deadline. SCREEN V3 (2026-10-08 00:25 KST, astra code review fixes): teacher lineage validated independently of the spec; persistent duplicate registry artifacts/experiments/SUBMISSION_SCREEN/fingerprints.json (missing/unreadable → HOLD; match with a COMPLETE submission → BLOCK); a submission_ref missing from the cache → HOLD; decision.basis must be exactly public|oof; back-test V3: only swa16 changes (BLOCK → PUBLIC-ONLY with the completed spec). After the 2026-10-08 submissions, re-screen rad5/coat5rad5/flight5/swa16 with their submission refs so the registry blocks exact twins; 17 pre-screen submissions remain outside the registry. SCREEN V5 (2026-10-08 01:38 KST): coverage reconciled against the submissions cache (all 21 non-ERROR submissions covered: 17 acknowledged historical entries in artifacts/experiments/SUBMISSION_SCREEN/historical_submissions_v1.json + 4 screened); checkpoint identity no longer depends on arm names; replicate and per-fold label quarantine checks; screening WITHOUT a --submissions cache now HOLDs. AFTER THE 2026-10-08 SUBMISSIONS: refresh the cache (kaggle competitions submissions --csv), then `tools/submission_screen.py --bind-submission <rad5|coat5rad5|flight5|swa16> <REF> --submissions <cache>` for each. DATES CONFIRMED (2026-10-08 12:48 KST, consult): A = unchanged anchor 56773226, never dropped (a 0.94 own model is still below 0.943 and the medal region); B = strongest validated complementary candidate, submitted by 10-18, COMPLETE by 10-20; OAI-derived public checkpoints (D-B-058) are excluded from both. REVIEW (2026-10-08 17:03 KST): A stays the anchor 56773226 (both reviewers; the hedge if an OAI ruling removes the 0.950 crowd); verify the anchor's lineage is OAI-free (D-A-050 leaves v5 provenance unknown — fable); B = B3 provisionally; avoid two near-identical variants; reconcile the old OAI exclusion text with the user's later OAI decision. SUBMISSION-VALUE RULE (2026-10-08 19:48 KST, user: 'low-scoring submissions — still meaningful when our ruler is fairly accurate?'): a public reading is spent only on (a) final-selection candidates and (b) changes the ruler cannot measure (all-data / no-OOF models, label-source changes, cross-family or OAI blends); within-family training changes are judged on the ruler alone after one confirmatory reading per new effect size. 10-09 plan: B3 (with the user's approval), swa16 (all-data, H-B-63 go/no-go), EXP140 (one confirmatory ruler reading of the largest delta, +0.009); flight5 v3 DROPPED (no decision depends on it). FINAL SUBMISSION PLAN (2026-10-08 19:55 KST, decision consult fable + astra, both read-only; user asked to save submissions): 10-09 after 09:00 KST — EXP140 (AUG1 RadR50 2-seed×5+TTA, non-OAI; registered 0.938–0.941, ≤0.932 = no transfer) and swa16 (all-data; ≥0.923 supports H-B-63/H-B-54, ≤0.914 stops, between = hold); then ONE OAI-line blend (user-approved): B3-AUG1 if its Public3 dry run passed and EXP140 reads ≥ 0.933, otherwise B3 (if EXP140 ≤ 0.932 or unscored by 15:00). Never both twins (partner Δproxy +0.0004 ≪ public noise ±0.007–0.010). Skipped: flight5 v3, any B3 partner-swap successor, routine bundle/external readings (ruler-judged after EXP140 confirms). Later budget: ≤ 3 more (H-B-63 standalone only if swa16 ≥ 0.923; ONE structurally different successor blend with a new approval by 10-16; one reserve), all COMPLETE by 10-18–10-20. Selections: A = anchor 56773226 (replaced only by a credibly stronger COMPLETE non-OAI model); B = the 10-09 OAI blend (replaced only by the pre-registered successor if public ≥ B and Gold58 direction positive). If the host rules against OAI: B = EXP140 if ≥ 0.942, else the best other non-OAI COMPLETE. Agreement: one OAI submission, EXP140 + swa16 yes, flight5 no. Difference: astra budgeted 6–8 further readings incl. a bundle and an external-data bundle; fable ≤ 2–3 → adopted fable's tighter budget (user's priority to save submissions; submission-value rule).

### H-B-36 — Few-parameter report-state calibration on Gold58 for Effusion and Baker's (conditional on the label test)
- Sources: D-B-013, D-B-014, D-C-001, D-C-003
- Hypothesis: Mapping report states (small / moderate / large / ungraded / not addressed) to official-positive probabilities estimated on Gold58 for Effusion and Baker's only — two or three numbers per target — improves the relabel_v1 training targets where the strict thresholds under-call.
- Evidence: the one-time Gold58 diagnostic (handoff 13:22): of 35 official effusion positives the strict labels called 9 pos, 15 unc, 11 neg (reports describing small/trace effusion that the official read graded moderate–large); Baker's 0.819 vs flight 0.944; a single structural rule (Synovitis not addressed → effusion) raised gold Synovitis 0.678 → 0.790 in the forum (D-B-013). DUPLICATE CHECK (18:43 KST): no calibration item in plan.md or discoveries.md; H-A-06 (slot A) audits label missingness/uncertainty definitions from source data, not a Gold58 calibration; D-C-003 found no lost positives for MCL/menisci/fracture, which this item does not touch.
- Improvement: uses Gold58 only for a pre-registered, few-parameter mapping on two targets with leave-one-out inside Gold58, instead of free tuning; judged on public.
- Impact: 1
- Information: 1
- Confidence: 1
- Unblock: 0
- Diversity: 1
- Cost: 1
- Priority: 7
- Resource: CPU
- Parallel: YES
- Other: NONE
- Next test: only if the three-arm label test (H-B-27/H-B-28 joint reading) moves public by ≥ 0.009; then estimate the state → probability map with leave-one-out on Gold58, apply it to relabel_v1 for the two targets, retrain CoAtNet5, one public submission (pre-register the bar before training). UPDATE (2026-10-08 19:53 KST): extend from relabel_v1 to the AUG1/bundle training targets; estimate the few per-target calibration numbers by 2-fold cross-fitting on Gold58 (29/29) to bound overfitting; priority targets Effusion, Baker's, Synovitis (largest report-vs-official gaps, D-B-014/D-C-001).

### H-B-37 — Multimodal LLM reading of report + key MRI slices on Gold58 for Effusion and Baker's (low-priority pilot)
- Sources: D-B-013, D-A-022, D-B-029
- Hypothesis: A Claude sub-agent shown the report together with a few sagittal fluid-sensitive slices grades Effusion and Baker's size closer to the official read than the report alone, enough to rival our trained image models on those two targets.
- Evidence: report sizes under-state the official grade for these two size-defined, visually large findings (handoff 13:22); our trained image models are already strong on them on Gold58 (≈0.93–0.98, D-A-022), so the bar is high; the forum's 0.950 VLM was a fine-tuned Qwen LoRA, not zero-shot reading (D-B-013); commercial LLM use is allowed under reasonable cost (D-B-029), which rules out whole-corpus image reading. DUPLICATE CHECK (18:43 KST): no zero-shot image-reading item or result in plan.md or discoveries.md.
- Improvement: a 58-study, two-target pilot (≈0.5–1 M tokens) that compares against report-only labels and the fully held-out 5-fold image models before any larger use.
- Impact: 1
- Information: 1
- Confidence: 0
- Unblock: 0
- Diversity: 2
- Cost: 1
- Priority: 7
- Resource: CPU
- Parallel: YES
- Other: NONE
- Next test: only after H-B-36 or if the user asks; render 3–5 sagittal fluid-sensitive slices per Gold58 study to PNG locally (no upload), one sub-agent grades the two targets with report + images; pre-registered: continue only if its Gold58 AUC on both targets is at least the 5-fold image models' AUC on the same studies.

### H-B-38 — Final models trained with the 58 official labels added (final week only)
- Sources: D-B-013, D-B-014, D-A-002, D-A-023
- Hypothesis: Retraining the chosen final own stack on the 4,349 report-labelled studies plus the 58 studies with official image-read labels (fixed up-weight) moves the models slightly toward the official grading thresholds without harming them.
- Evidence: train.csv supplies official labels for 58 studies (D-B-013); report-derived sets reach only ≈0.88–0.90 against them (D-B-014), so they are the only supervision in the grader's own definition; they are 1.3% of the data, so the expected effect is small; using them for training removes the only local official-label diagnostic (D-A-002), and an old full-data mode used Gold for checkpoint selection, which stays forbidden (D-A-023). Proposed to the user when they asked whether we could add the 58 studies ourselves. DUPLICATE CHECK (19:47 KST): no plan item or discovery trains on the Gold58 labels; H-B-26 keeps Gold out of every screen and H-B-36 uses Gold58 only to estimate a calibration map.
- Improvement: uses Gold58 as training data only after every decision that needs it as a diagnostic is made; the up-weight is fixed in advance, never tuned.
- Impact: 1
- Information: 1
- Confidence: 1
- Unblock: 0
- Diversity: 1
- Cost: 1
- Priority: 8
- Resource: GPU
- Parallel: NO
- Other: Kaggle (Public3 + one submission)
- Next test: in the final week, after H-B-35's selection rule has picked the own stack: retrain that stack with the 58 official-label studies added at a fixed weight ×4 (each counted four times; fixed now, 19:47 KST) and with no Gold-based selection; one public submission against the same stack without them; pre-registered non-inferiority: keep the Gold-trained version if public ≥ (without − 0.003), otherwise keep the version without Gold. REVIEW (2026-10-08 23:10 KST, fable + astra): DISAGREEMENT — astra: add Gold58 (×5 ≈ 6 % of training weight) to ONE final candidate after the recipe is frozen, keep its Gold-excluded predecessor, reconcile this item's ×4 with H-B-54's ×5; fable: no (≤+0.002, costs the only official yardstick and a slot). Decision deferred to the final-fit freeze (≈10-15) with a decision consult; nothing changes now.

### H-B-39 — ON HOLD (evidence, 21:37 KST): Colab lane for RadR50 5-fold on relabel_v1 labels
- Sources: D-B-030, D-B-008, D-B-014, D-A-022
- Hypothesis: A RadImageNet ResNet50 centre-slice 5-fold trained on the strict relabel_v1 labels adds a decorrelated member to the own ensemble (label diversity × our best family), whatever the single-label-set public result is.
- Evidence: RadR50x5 is our best OOF family (0.8914, D-B-030); report-label sets disagree strongly (53% of studies change ≥ 1 decision between Dread and relabel_v1), so models trained on them are decorrelated members; label-set ensembling is the hedge when no local instrument can rank label sets (D-B-014). Colab trajectories cannot be paired with local arms (D-B-008), so this is a member, never a paired arm. The user asked at 20:00 KST to start using Colab. DUPLICATE CHECK (19:56 KST): no RadR50 × relabel_v1 run exists or is queued; H-B-34 listed "label sets" last on the local GPU — moved here; H-B-29 covers the CoAtNet label-set ensemble only.
- Improvement: uses the idle Colab lane with per-epoch checkpoint/resume (earlier Colab runs were lost to reclaims) and a local poller that keeps every checkpoint.
- Impact: 2
- Information: 1
- Confidence: 2
- Unblock: 1
- Diversity: 2
- Cost: 1
- Priority: 13
- Resource: GPU
- Parallel: YES
- Other: Colab
- Next test: a CPU sub-agent builds the resumable runner, Colab bootstrap v3, local poller and runbook (docs/runbooks/colab-exp114.md); the USER creates the Colab session and uploads the Kaggle token (credential step); then folds 0–4 run on the T4 (≈75 min per fold); assemble OOF; scored against Dread, flight and relabel_v1; enters H-B-29/H-B-35 as a member candidate only. ON HOLD (20:16 KST; source: the user,; reason: the Colab path needs a resumable runner, a local poller and a user credential step for every session; evidence: the builder sub-agent was stopped before writing any file): no Colab work; the RadR50 × relabel_v1 run moves back to the local GPU as the last step of H-B-34 (tools/exp113_radr50_local.py already supports --label-set relabel_v1, ≈1.7 h). Resume only if the user asks. RESUMED (21:16 KST; source: the user): hold lifted; the builder sub-agent restarted with the same specification; RadR50 × relabel_v1 leaves the local H-B-34 order again; the user uploads the Kaggle token once per session. ON HOLD AGAIN (21:37 KST; source: the free Colab session knee-radr50-rl was created at ≈21:20, the user uploaded the token, the corpus prefetch started, and by ≈21:37 the server reported no active session (drivemount 404, colab status: not found) — a second loss after the 2026-10-06 22:02 loss; reason: every reclaim needs a fresh session and a user token upload, which is not workable at this rate; evidence: CLI outputs in the 21:37 handoff event): the builder sub-agent was stopped; RadR50 × relabel_v1 returns to the local GPU (EXP113 --label-set relabel_v1, ≈1.7 h) as the last H-B-34 step; resume only on a paid/stable runtime or at the user's request.

### H-B-40 — Add the public Raptor v9 single model (CC0, outside the anchor) to the own stack
- Sources: D-B-034, D-B-019, D-A-050, D-B-030
- Hypothesis: A rank mean of CoAtNet5 + RadR50x5 + public Raptor v9 (each family weighted equally) scores higher on public than CoAtNet5 + RadR50x5 alone, because v9 alone reaches 0.927–0.929 and is a different model and input (80 slices, 384 px) from both.
- Evidence: v9 alone measured at public 0.927 (k24) – 0.929 (k62) by another participant with 64-slot / 224-px-cache serving (author report, earlier speedprobe versions), author 0.932 (D-B-034, D-B-019); it is not part of the 0.943 anchor (the anchor runs v5, v10, v8; D-B-018); weights are CC0 and local; it was trained on all data with likely Gold-based epoch selection (D-A-050), so no OOF on our folds exists and the decision is public-only. DUPLICATE CHECK (20:05 KST): no plan item or discovery uses the public v9 weights as a member; D-B-023 tested our own 80-slice training (different question); H-B-32 blends the whole anchor, not v9.
- Improvement: adds the strongest licence-clean public single family to the own stack, independent of the anchor, with a fixed equal-family weight.
- Impact: 3
- Information: 2
- Confidence: 2
- Unblock: 1
- Diversity: 1
- Cost: 2
- Priority: 16
- Resource: CPU
- Parallel: YES
- Other: Kaggle (Public3 + one submission)
- Next test: after H-B-31's two scores (2026-10-08): choose the serving contract first (21:42 KST, D-B-034 update): the cheapest faithful replication is the speedprobe's 64-slot K24–K62 path behind 0.927–0.929 (our 336-px 64-slot cache may not match it byte for byte); the native 80-slot/K78 path is a different arm scored only by its author; then build a release arm for v9 (D-B-019/D-B-023 code), with a manual lineage check because the screen would PASS a mis-bound receipt, CPU parity on a few studies, runtime screen, then ONE submission of CoAtNet5 + RadR50 + v9 equal family rank mean, screened by tools/submission_screen.py (lineage will flag v9 as trained on all data). Pre-registered (20:05 KST): kept as the own stack if public ≥ (CoAtNet5 + RadR50 score) + 0.005; dropped if ≤ that score. DIVERSITY NOTE (2026-10-08 01:47 KST, user question): v9 is outside the anchor but in the SAME Raptor/CoAtNet-2 family as the anchor members v5/v8/v10, so its marginal value inside an anchor blend is likely small (the 0.8/0.2 blend with our CoAtNet-family arm scored 0.942, D-B-031); priority stays below RadR50 and genuinely different families (H-B-43 VLM, H-B-47 source heads); measured own-model rank correlations (mean per-target Spearman, 4,343 OOF): CoAtNet5–RadR50 0.819, CoAtNet5–ResNet34 0.812, RadR50–ResNet34 0.794, same-recipe RadR50 local vs Kaggle 0.936; anchor correlation cannot be measured locally (no OOF).

### H-B-41 — Faster serving preparation (DICOM header whitelist + parallel prefetch)
- Sources: D-B-034, D-B-005, D-B-018
- Hypothesis: Reading only the needed DICOM header tags and prefetching studies in parallel cuts ordering-metadata preparation per study by about 2× without changing any model input byte.
- Evidence: reported 1.701 → 0.803 s per study for header reads only, ≈2.1× on that stage, not pixel decode (D-B-034 updated 21:42 KST; the earlier "3.0 → 0.8" merged two numbers); our releases decode every study once per arm and the combined anchor + own kernel projects 7.6–7.95 h of the 9 h budget (D-B-018), so preparation time limits how many members fit. DUPLICATE CHECK (20:05 KST): D-B-018's shared-decode adapter removes repeated decoding across arms but not header parsing or prefetch; no item covers this.
- Improvement: frees serving time for more members (H-B-32, H-B-40) with an input-byte equality check as the gate.
- Impact: 1
- Information: 1
- Confidence: 2
- Unblock: 2
- Diversity: 0
- Cost: 1
- Priority: 9
- Resource: CPU
- Parallel: YES
- Other: NONE
- Next test: implement in the release decode path, prove byte-identical model inputs on 100 studies (hash_stage), measure s/study locally and in the next Public3.

### H-B-42 — Ruler test bench: find the local metric whose gains predict public gains (all scored own submissions, candidate references incl. a Gold58-calibrated soft-gold reference), then make it the screen metric
- Sources: D-B-035, D-B-033, D-B-016, D-B-020
- Hypothesis: Consensus-filtered OOF (pairs where independent report-label sets agree) predicts public differences better than single-reference OOF, including for label-set changes that single references cannot judge.
- Evidence: consensus labels are right ≈90% vs ≈82% on Gold58 and are neutral between label sets (D-B-035); it rescored relabel_v1 vs Dread at +0.0026 where the own references give −0.0082 / +0.0103; the user asked for a local way to validate what currently only public can judge. DUPLICATE CHECK (20:10 KST): D-B-014 averaged label probabilities (no gain on Gold58), a different operation; no consensus-evaluation item exists. Impact raised 2→3 at 22:40 KST (user direction): the ruler gates every later decision.
- Improvement: turns the label-validation problem from "public only" into "local estimate on agreed cases + one public check", and tests that claim on submissions already planned.
- Impact: 3
- Information: 3
- Confidence: 2
- Unblock: 2
- Diversity: 2
- Cost: 0
- Priority: 21
- Resource: CPU
- Parallel: YES
- Other: NONE
- Next test: record each prediction before its public score (relabel_v1 tonight: within 0.914 ± 0.007; CoAtNet5 + RadR50 and RadR50 alone tomorrow: consensus deltas +0.0114 and +0.0064 over CoAtNet5; flight_v4 once its 5-fold exists); after these four scores, compare the hit rate and the |prediction − public| of consensus OOF vs Dread OOF; if consensus is not worse, add it to tools/submission_screen.py as a REPORTED DIAGNOSTIC line, not an automatic gate (fable + astra 21:05 review, adopted 21:54 KST: consensus excludes disputed pairs, keeps shared report errors, Synovitis agreement 0.709, and four correlated public scores cannot validate it as a gate); any public "adopt" reading from the 2026-10-08 scores stays provisional until consensus OOF or a replicate agrees (fable). PREDICTION LOG (21:12 KST, before any of these scores exists): consensus-pair macro AUC (Dread, Steven v4, Pilkwang v2, relabel_v1 agree; 4,343 studies) of the three CoAtNet5 label-set models — Dread 0.9445, relabel_v1 0.9471 (+0.0026), flight_v4 0.9397 (−0.0048); predicted public: relabel_v1 0.914 ± 0.007 (56907117, pending), flight_v4 0.909 ± 0.007 (to be submitted after the reset); own-reference columns are home-biased (each model is best on its own labels) and reference difficulty differs (positive rate relabel 13.8%, Dread 18.7%, flight 26.9%), so only the consensus column is compared. DATA POINT 1 (22:04 KST, D-B-038): relabel5 public 0.913 vs predicted 0.914 ± 0.007 (consensus) — hit; own-reference predictions ≈0.906 (Dread) and ≈0.924 (relabel_v1) missed. SECOND PREDICTOR (registered 22:34 KST, before the 2026-10-08 scores; D-B-041): public ≈ 0.914 + Dread-OOF Δ / c. Level-consistent c ≈ 0.94 (OOF 0.8875 vs public 0.914) → CoAtNet5+RadR50 ≈ 0.926, RadR50 alone ≈ 0.918; the research sub-agent's c ≈ 0.6 → ≈ 0.933 and ≈ 0.920. Not applied to flight5 (a label change, not class-conditional noise). Add a cross-set column (training label ≠ reference, hengck23) to every comparison. USER DIRECTION (22:39 KST): build an accurate ruler WITHOUT extra submissions before running label interventions, so that a local gain can be trusted to transfer. Bench (sub-agent started 22:39 KST, output .local/ruler-bench-20261007/): ≈12 scored own submissions with OOF (public 0.893–0.917) × rulers (Dread, each silver set, consensus pairs, soft-gold = P(official | silver labels) fitted on Gold58 with leave-one-study-out, Gold58 itself); judged by Spearman, pairwise sign agreement and leave-one-out error; predictions for the three 2026-10-08 submissions registered before their scores. H-B-46/H-B-47 wait for the bench verdict (their judging rule is rewritten to use the validated ruler where it applies). BENCH RESULT (23:07 KST, D-B-042): no ruler validated by the pre-registered rule (n = 9); post hoc, Dread 5-fold delta transferred ≈1:1 within one protocol/label set; provisional ruler = Dread delta ≥ +0.004 with CI > 0 and consensus delta ≥ 0, like-for-like controls only, label changes judged by consensus. Next: when the 2026-10-08 scores arrive, add the three points to the bench (bench.py --add) and re-run; the Dread headline is refuted if coat5rad5 ≤ 0.914 or rad5 is outside [0.908, 0.927]. BENCH POINTS (2026-10-08 12:41 KST): rad5 0.925 (forecast 0.918 [0.908, 0.927], hit, low by 0.007), coat5rad5 0.926 (forecast 0.925 [0.916, 0.935], hit) — D-B-049/D-B-057; cross-family additions transfer far less than OOF suggests (2 of 2). FORECASTS BEFORE EVERY SUBMISSION (2026-10-08 12:53 KST, user question on ruler accuracy): 2/2 pre-registered forecasts hit; keep registering a forecast before each score (flight5 0.909 [0.895, 0.923] and the swa16 rules are on file for 10-09); same-family RadR50 changes are forecast as 0.925 + pooled ΔDread. STANDING LEDGER (2026-10-09 01:3x KST, user rule): tools/evidence_ledger.py v1.0.0 (2 tests) — every submission logs all panel sources before scoring and its public reading after; 24 historical bench points imported (V888/C529/G4343: report-label sources all Spearman ≈0.74–0.87, Gold58 too few points); first prospective entries: H-B-34-exp140-aug1 (G4343: dread .9001, consensus .9568, flight .8585, relabel_v1 .9206, jev_hybrid .8880), H-B-58-b3 (Gold58 .9274, exposed). Next: add each reading; review the report at every six-hour review.

### H-B-43 — Vision-language model pilot (Qwen-class ≈2B VLM + LoRA, 384 px) as a decorrelated member: feasibility first, then one fold
- Sources: D-B-013, D-B-034, D-B-008
- Hypothesis: A small open vision-language model fine-tuned with LoRA on multi-slice inputs reaches at least our CoAtNet level on fold 0 and, being a different model family, adds more to the own ensemble than another CNN.
- Evidence: a forum author reports a Qwen3.5-2B VLM with LoRA at 384 px, single fold, public 0.950 in ≈3.5 h of training (D-B-013; self-report, no code seen); every other reported 0.95 single used ordinary CNNs with better supervision; T4 has no bf16, and a bf16-trained large model served in fp16 risks overflow (numerical-parity skill); the 9 h serving budget is shared with other members. astra proposed a small image-language LoRA pilot at the 17:05 review; it was deferred until H-B-33/H-B-34 report; the user asked whether it is worth trying. DUPLICATE CHECK (21:00 KST): no plan item or discovery; D-B-013 holds only the report; H-B-37 is a zero-shot LLM reading pilot on Gold58, a different question.
- Improvement: a staged, kill-criteria-first pilot instead of an open-ended build.
- Impact: 3
- Information: 2
- Confidence: 1
- Unblock: 0
- Diversity: 3
- Cost: 3
- Priority: 14
- Resource: GPU
- Parallel: NO
- Other: NONE
- Next test: (1) tonight, CPU/web only: recover the author's method (input layout, head, loss, LR, epochs, GPU, time) from the forum, confirm the exact open model, its licence and size, estimate memory and time on the RTX 4070 Ti SUPER (16 GB) and fp16 serving time on T4 for 1,300 studies; (2) only if feasible: fold 0 pilot on the local GPU after the current chain, pre-registered stops — consensus-OOF (D-B-035) more than 0.01 below CoAtNet fold 0, projected T4 serving > 3 h, or any fp16 non-finite output on 100 studies. FEASIBILITY (21:12 KST; read-only sub-agent, report .local/vlm-feasibility-20261007/vlm_feasibility.md): GO conditional — author (NguyenThanhNhan, topic 735304) used Qwen 3.5 2B, 384 px, LoRA on the language model with the vision encoder fine-tuned jointly, single fold 0.950, ~3.5 h (unclear whether training); all other details unknown, no code; Qwen/Qwen3.5-2B is Apache-2.0, 2.27 B parameters, 4.55 GB bf16, 144 tokens per 384 px image, needs transformers ≥ 5.2 + peft (separate venv); estimated 11.5–13 GB of 16 GB with gradient checkpointing, 0.8–1.6 h per epoch on the local GPU; T4 serving for 1,300 studies 10–20 min fp16 or 34–41 min fp32; a MedGemma fine-tune was reported unhelpful by another participant. WAITING for the user to approve the 4.55 GB weight download and the new venv; proposed amendment (to be fixed before any run): an fp16 non-finite output switches serving to fp32 (fits the budget) instead of stopping. APPROVED (21:23 KST): the user approved the weight download and the separate venv; the fp16 rule is amended BEFORE any run — a non-finite fp16 output switches serving to fp32 (34–41 min for 1,300 studies on T4) instead of stopping; the other stops stand (consensus-OOF more than 0.01 below CoAtNet fold 0; projected T4 serving > 3 h). Environment: .venv-vlm (system-site-packages; torch 2.10.0+cu126 because the local driver 561.17 supports CUDA ≤ 12.6 and transformers 5.x needs torch ≥ 2.6; transformers 5.x, peft); weights storage/weights/qwen__Qwen3.5-2B; tooling being built by a CPU sub-agent (EXP115); GPU slot after EXP113 F4 (≈00:30). FALLBACK ORDER (2026-10-08 03:06 KST, astra survey .local/vlm-survey-20261008/, no live web — specs marked unverified): keep Qwen3.5-2B through its 48-step gate (only competition evidence: 0.950 self-report); if too slow or failing → Qwen3-VL-2B (Apache-2.0, standard attention, no DeltaNet kernels needed, est. 10–30 min T4 serving) → InternVL3.5-1B/2B after licence verification → SmolVLM2-2.2B; Qwen3-VL-4B only if 2B shows useful validation/complementarity; Qwen2.5-VL-3B (Qwen Research License), Gemma 3 (Gemma terms), MedGemma (HAI-DEF terms; one participant's "didn't help much", post 3533150), Kimi-VL (16B MoE) and ≥7B medical VLMs not preferred. Any processor/slice-packing change needs a new registered protocol; supervision stays fixed. G0 GATE FAILED (2026-10-08 10:48 KST, D-B-053): Qwen3.5-2B 1.44 s/micro-step, projected 4.44 h; next: EXP123 Qwen3-VL-2B pilot (user allowed the download 11:0x KST; sol building new files). ON HOLD (2026-10-08 17:03 KST, direction review: fable 'stop — exclusive GPU, no partner path in 14 days', astra 'keep last'): removed from the GPU chains; revisit only if a lane is idle after the bundle and the external screens.

### H-B-44 — Consensus-weighted supervision: RadR50 trained on Dread with lower loss weight where the report-label sets disagree
- Sources: D-B-035, D-B-036, D-B-011, D-B-030
- Hypothesis: Training RadR50 on Dread labels with per-(study, target) loss weight 1.0 where the four report-label sets agree and 0.4 where they disagree (pos_weight fixed to Dread's) raises Dread and consensus OOF over the unweighted local control.
- Evidence: where the independent sets agree, the label matches Gold58 89.6% of the time versus 81.7% for one set (D-B-035; 79% of the 4,343 study-target pairs agree); top-20 participants name labels as the main lever (D-B-036); recomputed class weights confound label tests (D-B-011), so pos_weight stays Dread's; proposed by fable in the 21:05 review. DUPLICATE CHECK (21:54 KST): H-A-06 confidence weighting was taken over as H-C-04 and closed as an audit without a training test, and C's discoveries ask for new independent evidence before a supervision intervention — D-B-035's Gold58 agreement accuracy is that new evidence; H-B-28 replaces labels and H-B-29 ensembles label-set models, neither weights pairs.
- Improvement: uses cross-set agreement as a per-pair confidence instead of replacing labels; prevalence and pos_weight unchanged, so only the weighting differs from the EXP113 control.
- Impact: 2
- Information: 2
- Confidence: 1
- Unblock: 0
- Diversity: 2
- Cost: 2
- Priority: 14
- Resource: Kaggle T4 + Vast
- Parallel: NO
- Other: NONE
- Next test: (updated 2026-10-09 13:42 KST, Taeyang resource push + D-B-081) EXP150 on the AUG1 RadR50 recipe, fold 0: seed-42 pair (control, weighted) in ONE Kaggle 2×T4 session, seed-43 pair on the Vast RTX 3090 (each pair on one GPU type; identical initial-state FILE everywhere); sol builds it (13:42). Registered reading changes because consensus is biased by construction for this arm (same four sets): primary = mean of JEV HYBRID and aquila_v1 deltas > 0 in both pairs, Dread guard ≥ −0.003; consensus report-only. Supersedes the 21:54 EXP113-based rule (written before AUG1, JEV and aquila existed).

### H-B-46 — Targeted teacher fill: replace only report-silent or disputed cells with calibrated true-OOF predictions (teacher-heavy), RadR50 on Dread otherwise
- Sources: D-B-041, D-B-027, D-B-035, D-B-040
- Hypothesis: Training RadR50 on Dread labels where report-silent cells (Synovitis and PF OA where relabel_v1 says "not mentioned") and Effusion cells where the four sets disagree are replaced by 0.3·Dread + 0.7·calibrated CoAtNet5+RadR50 true-OOF probability raises public over the unchanged RadR50x5.
- Evidence: the only reported label gains in this competition come from filling report-silent cells with true OOF predictions, teacher-heavy (topic 735304 post 3526342), and per-source teachers 0.938→0.949 (3524699); global 50/50 mixing did not transfer for us (distill_v1 0.912 vs 0.914, D-B-027) nor for Nicolai Karcher (3529062); our teacher outputs are scale-distorted by pos_weight (D-A-047), so calibration comes first; report-label errors are shared across sets (D-B-040), so image-side information is the only independent source. DUPLICATE CHECK (22:34 KST): D-B-027 is the global 50/50 form; D-C-005 designed the leak-free fold structure (reuse it); H-B-34's "leak-free distillation" step is replaced by this item; Baker's and Medial OA stay untouched (silence ≈ negative on Gold58, topic 733932).
- Improvement: local and teacher-heavy instead of global 50/50; calibrated teacher; cells chosen by report state and cross-set disagreement, not by model confidence.
- Impact: 2
- Information: 2
- Confidence: 2
- Unblock: 0
- Diversity: 1
- Cost: 2
- Priority: 12
- Resource: GPU
- Parallel: NO
- Other: NONE
- Next test: CPU first (sub-agent with tests): calibrate the CoAtNet5+RadR50 OOF per target on Dread (Platt within training folds only), write the filled label file with a cell-mask receipt and the D-C-005 leak-free fold rule (each fold's teacher never saw that fold's validation studies), pos_weight fixed to Dread's; then RadR50 5-fold on the local GPU (≈1.7 h) after EXP113; judged by ONE public submission (Dread and consensus OOF cannot judge a label intervention — consensus excludes exactly these cells). Pre-registered (22:34 KST): adopt if public ≥ RadR50x5's own public + 0.006; reject if ≤ −0.003. ORDER (22:39 KST, user direction): starts only after the H-B-42 ruler bench reports; if a validated ruler exists, it replaces public as the first judge. TOOLING DONE (23:27 KST): tools/exp116_teacher_fill.py + runner + 55 tests; label files storage/labels/derived_teacher_fill/ (one file per student fold, fold-k rows stay Dread; calibrators fitted outside folds j,k; receipt 334f2cf8…); fills: PF OA 899 cells (relabel == 0.05; ≈19% are report-negated cells scored 0.05, inseparable without raw states), Synovitis 3,725, Effusion 1,737 (40% of cells); student OOF quarantined as EXP116_TEACHER_FILL_OOF_LEAKY (decision_eligible false, D-B-024) — the judge is public only. OPEN before admission: (a) calibration scale — rint-calibration drops the Synovitis target mean 0.449→0.312 while pos_weight stays 1.23 (fill-implied 2.20), a scale confound; B asked the builder for a soft-scale calibration variant (mean-preserving on filled cells) as the default; (b) teacher includes RadR50x5 (same family as the student); (c) not admitted to the GPU chain yet (user direction: ruler first; D-B-042 says label changes have no validated local ruler, so this item costs one public submission — decide after the 2026-10-08 scores). CALIBRATION (23:39 KST): tools/exp116_teacher_fill.py v1.1.0 makes soft-scale calibration the default (Platt against soft Dread, calibrated mean = soft-Dread mean on fitted rows, |Δ| < 1e-12); target levels now stay at Dread (Synovitis 0.449 → 0.451 vs 0.312 under rint); rint kept as label set teacher_fill_v1_rint for the record; 63 tests pass. Trade-off: the soft Synovitis fill is low-contrast (Platt slope 0.30, mean |Δ| 0.073 on filled cells, 104 flips vs 284) — if H-B-46 shows no effect, weak injected information is a candidate reason; a rank-preserving spread at the soft mean would be a separate variant (not built). ASTRA REVIEW (23:54 KST, .local/astra-20261007/astra-codereview-out.md): (1) BLOCKER as specified — student fold k's labels contain fills from teacher j, which trained on fold k, so the "no fold-k ancestry" rule is NOT met; B's decision: keep the design for a PUBLIC-only judgement (the hidden test is untouched by any teacher; the student OOF is quarantined as LEAKY and never used), and state this explicitly instead of claiming leak-free OOF; nested teachers (≈3 GPU h per outer fold) only if a later decision needs H-B-46 OOF. (2) MAJOR — soft calibration preserves the mean on fitted rows, not on filled cells: PF OA filled-cell mean 0.144→0.238 and whole-target 0.357→0.376 (Effusion +0.010); registered now as part of the intervention (the fill raises PF OA/Effusion levels slightly), not a mean-preserving fill. SUPERSEDED REQUIREMENTS (2026-10-08 00:05 KST, astra re-review): the earlier wording of this item that asks for a leak-free student OOF and a mean-preserving fill is superseded — the student OOF is LEAKY by design and diagnostic-only (never a ruler, gate, ensemble weight, stacking input or teacher), the judgement is ONE public submission, and the soft calibration's whole-target level shift (PF OA +0.019, Effusion +0.010) is part of the registered intervention; a CPU sub-agent is propagating this to the tool text, receipt/manifest (new versions) and adding DIAGNOSTIC_ONLY markers enforced by g1_compare.py and exp103_distill_labels.py.

### H-B-47 — Label-source heads: one RadR50 trunk with four heads (Dread, relabel_v1, Steven v4, Pilkwang v2), served as the head rank mean
- Sources: D-B-041, D-B-038, D-B-040
- Hypothesis: A shared trunk with one head per label source, served as the equal rank mean of the heads, beats single-source RadR50 on public while costing one arm at serving time.
- Evidence: separate heads for curated and noisy data gave public +0.021 in Freesound 2019 (Akiyama & Sato, DCASE 2019 report) and per-annotator heads beat consensus labels (Guan et al. 2017); our label sets disagree by definition (relabel: own reference +0.010, Dread −0.008, public −0.001, D-B-038); H-B-29's three separate CoAtNet arms do not fit the 9 h serving budget (H-B-29 runtime note), a multi-head single arm does. DUPLICATE CHECK (22:34 KST): no plan item or discovery uses source-specific heads; H-B-29 ensembles separate models.
- Improvement: label diversity at single-arm serving cost; the trunk sees all sources, heads absorb source-specific definitions.
- Impact: 2
- Information: 2
- Confidence: 1
- Unblock: 1
- Diversity: 2
- Cost: 2
- Priority: 13
- Resource: GPU
- Parallel: NO
- Other: NONE
- Next test: CPU first (sub-agent with tests): add a `--heads` option to the EXP113 runner (summed per-head BCE with each source's own pos_weight; rows missing a source are masked for that head), fold-0 smoke; then a fold-0 matched screen against EXP113 F0 on Dread, consensus and the cross-set column; continue to five folds only if the head mean is not below EXP113 F0 by more than 0.002 on any reference; judged by ONE public submission. Pre-registered (22:34 KST): adopt if public ≥ RadR50x5's public + 0.006. ORDER (22:39 KST, user direction): starts only after the H-B-42 ruler bench reports; if a validated ruler exists, it replaces public as the first judge. TOOLING DONE (23:27 KST): tools/exp117_multihead.py + runner + 30 tests; four heads copied from the EXP113 seed-42 head (initial state 397ed4aa…), masked per-head BCE with each source's own pos_weight, served output = per-target rank mean of the heads (each head's K24/K36 OOF saved); CPU smoke 2 updates OK. OPEN before admission: (a) the summed loss makes clip 3.0 bind every step (trunk step differs from EXP113) — consider clip scaled by the number of heads; (b) no serving path for a multi-head checkpoint in the release tools (needed before any submission); (c) pooled OOF mixes per-fold rank scales — compare per fold or re-rank pooled; (d) fold-0 screen only (its queue file lists all five folds; admit F0 alone); not admitted yet — after the 2026-10-08 scores. ASTRA REVIEW (23:54 KST): queue must be F0-only without failure continuation (fix in progress by a CPU sub-agent, with an explicit command for F1–F4 after a recorded screen pass); the "21 changed lines" test is being replaced by an ordered comparison; summed-loss clipping is a registered difference from single-head optimisation. QUEUE FIXED (2026-10-08 00:25 KST): queue-json defaults to F0 only, continue_on_failure false; F1–F4 only with --folds 1-4 --screen-pass <json> ({"verdict":"PASS","tag":"EXP117_MULTIHEAD_F0","protocol_sha256": current manifest}); manifest re-prepared, new sha 5ee284b1… (initial state unchanged 397ed4aa…); ordered runner-equivalence test. F0 SCREEN PASSED (2026-10-08 02:23 KST, D-B-047): Dread +0.0040 [+0.0004, +0.0083], consensus +0.0074; F1–F4 queued in chain V21 after EXP118 (continue_on_failure true: independent fits); next: pooled five-fold comparison vs EXP113, then a multi-head serving arm (or the Dread head alone) in release v7. FIVE FOLDS (2026-10-08 10:48 KST, D-B-051): head rank mean vs EXP113 Dread +0.0019 [−0.0001, +0.0040], consensus +0.0048 [+0.0030, +0.0070]; Dread head alone +0.0030; next: EXP122 seed replicates (s43, s44; being built) and a multi-head serving arm in release v7, then one public test with tools/kaggle_submit_once.py. PROSPECTIVE RULE (2026-10-08 11:10 KST, before any EXP122 output; D-B-051 update): seed-matched multihead−single deltas for seeds 42 and 43 both ≥ 0 on Dread, mean ≥ +0.004 with CI > 0, mean Δconsensus ≥ 0 → one public test; else stop the multi-head line. TIER-B RULE (2026-10-08 11:45 KST, user direction, registered before any EXP122 s43 output is evaluated; replaces the 11:10 rule): seed 42 pooled ΔDread +0.0019, share(Δ>0) 0.963, consensus +0.0048, flight +0.0157, relabel +0.0079 → provisionally Tier-B eligible; the multi-head replaces the single head in the next own-arm bundle if the seed-43 seed-matched delta (EXP122 s43 − EXP118) is ≥ 0 on Dread with no reference below −0.001; served with plane-aware flip TTA (D-B-056, also Tier B).

### H-B-48 — Colab single-model ablation loop run by the user (compact 12-slice ResNet34, one change at a time, four rulers)
- Sources: D-B-036, D-B-041, D-B-009
- Hypothesis: Single-model changes screened one at a time on fold 0 with two seeds against a matched control, read on four references (Dread, flight, relabel_v1, consensus), find augmentation/recipe changes whose gain exceeds the seed noise, and those changes also help RadR50.
- Evidence: rank-12 protocol (D-B-036): simple CNN, seeded CV, one-at-a-time augmentation ablations; augmentation is untested locally (the only test, D-B-009, was confounded by device/stack); user direction (22:56 KST): improve the single model in Colab together in chat, the rest per plan. DUPLICATE CHECK (22:56 KST): H-B-34's augmentation step is moved here (local GPU only ports confirmed changes); no other item screens augmentation.
- Improvement: matched control with two seeds and a measured noise floor; versioned notebooks (docs/learning/colab_knee_ablation_vX.Y.Z.ipynb); results saved to the user's Drive and read back in chat or from the open Colab tab (read-only).
- Impact: 2
- Information: 2
- Confidence: 2
- Unblock: 1
- Diversity: 1
- Cost: 0
- Priority: 15
- Resource: EITHER
- Parallel: YES
- Other: Colab (user)
- Next test: v1.1.0 run 1 = control seeds 42/43 + flip seeds 42/43 on fold 0 (≈1.5–2 h on T4). Reading rule (fixed now): a change is a candidate only if Δ on consensus > the control seed spread AND its 95% interval excludes 0 AND Dread/flight/relabel move the same way; then confirm on a second fold before porting to RadR50 (H-B-34). Each run's outcome becomes one discovery; each notebook change bumps the version. RULE AMENDED (23:07 KST, from D-B-042): for same-label changes (augmentation, recipe) the primary line is Dread Δ vs the fold-0 control of the same family — candidate if ΔDread > the control seed spread AND its 95% interval excludes 0 AND Δconsensus ≥ 0 AND flight/relabel do not move the other way; consensus becomes primary only for label changes. Fold-0 ResNet34 numbers are compared only with fold-0 ResNet34 controls (never with 5-fold ensembles). RUN 2 (planned 23:10 KST, user question on epochs): base-recipe check before more augmentations — control at 16 epochs (seeds 42/43, same OneCycle) vs the 8-epoch control; if better by the rule, 16 epochs becomes the control for later screens (flip re-checked only if it was borderline). Code: cell 7 v1.1.1 (PLAN tuples carry epochs). NOTEBOOK v1.2.0 (23:54 KST, astra review): results are reused only when the saved config (aug, seed, fold, backbone, epochs, bs, lr, data shape) matches — else ValueError; outputs written atomically; compare() skips variants whose seed set differs from the control's; component-level bootstrap not possible in the compact meta (fold 0: 888 studies / 873 components, minor). Cell 7 v1.2.0 = run 1 (ctrl, flip × seeds 42/43, 8 epochs) + run 2 (ctrl16 × seeds 42/43, 16 epochs). USER SCHEDULE (2026-10-08 00:03 KST): leave run 1 (v1.1.0) running untouched; when the user switches to v1.2.0, B starts 30-minute monitoring checks timed to the expected finish (read-only checks; reconnect + Run all on a drop; the Drive consent window needs the user's OK each time). NEXT STAGE after the v1.2.0 results: "<owner remark removed>" single-model strengthening along the rank-17 team's reported recipe (CoolinLai, team "Attention Is All You Knee": ResNet50 at 224 px, five folds, 0.954 public — a bare self-report, D-B-036): carry the confirmed augmentation/epoch choices to ResNet50, then five folds; designed with the user in chat. RUN 1 RESULT (2026-10-08 00:08 KST, D-B-043): flip rejected (negative on all four references); control spread 0.0004–0.0017; AUC still rising at epoch 8. Next run (v1.2.1, to be agreed with the user): ctrl16 (16 epochs) plus learning-rate screens 1e-4 and 1e-3 at 8 epochs, each two seeds; plane-aware flip needs slice-plane ids added to the compact meta first. REAL-MODEL NOTEBOOK (2026-10-08 00:18 KST, user request: "<owner remark removed>"): design v0 at .local/colab-v2-design/design_v0.md (EXP107 RadR50 5-fold baseline on the full 336 cache downloaded from the private Kaggle corpus dataset + candidate changes: 288 px, 24 epochs + EMA, K16, affine aug); astra design review running; a CPU sub-agent builds the config-driven resumable notebook (docs/learning/colab_knee_real_v2.0.0.ipynb via tools/learning/make_colab_real.py) with per-epoch Drive checkpoints, a time guard, release-compatible fold checkpoints and an embedded EXP107 OOF for like-for-like paired comparison; parity test against EXP107 checkpoints if available. The compact proxy loop (v1.x) is paused in favour of this (D-B-044). RECIPE DECIDED (00:24 KST, after astra design review .local/colab-v2-design/astra-design-out.md): EXP107 RadR50 unchanged except K16 training windows (single attributable change; 288 px, 24 epochs, EMA/SWA and affine dropped for lack of evidence and budget — EXP107 folds took 3,452–3,478 s each, ≈4.8 h; K16 ≈6.4 h); K36 primary eval; optional same-Colab fold-0 K12 control after the five folds; 9 h compute / 11.5 h wall limits; adoption gate ΔDread ≥ +0.004 with component-bootstrap CI > 0, Δconsensus ≥ 0, flight/relabel ≥ −0.002 vs embedded EXP107 OOF (benchmark, not matched control) → then ONE frozen public test; astra planning range 0.91–0.93 public, 0.95 not a defensible expectation for this notebook. NOTEBOOK v2.0.0 BUILT (2026-10-08 01:39 KST): docs/learning/colab_knee_real_v2.0.0.ipynb (407 KB, sha 61ebd5f4…; generator tools/learning/make_colab_real.py + tools/learning/real_v2_cells.py; CPU harness .local/colab-v2-build/): default mode screen (fold 0: K12 s42 → K16 s42 → K12 s43 → K16 s43, same-Colab paired component bootstrap, ≈4–5 h), production mode (K16 folds 0–4 + optional K12 control); CPU tests: tiny run, exact resume after clean stop and mid-epoch crash (trace + weights identical), parity with Kaggle EXP107 fold-0 OOF max |Δ| 9.1e-4 on 24 studies, centres/tensors 32/32, checkpoints pass exp104/exp111 checkpoint_meta, embedded OOF round-trip 2.4e-4; untested: Drive/token/Kaggle download and GPU-specific resume paths. astra pre-launch review running. COLAB NOTE (2026-10-08 12:53 KST): Colab ran out of free GPU after screen_k12_s42 and 14/20 epochs of screen_k16_s42 (Drive checkpoints kept); the K16 screen moved to the local GPU (EXP125); v2.0.1 crashes with a raw FileNotFoundError when the runtime has no GPU (nvidia-smi missing) — a friendly message belongs in v2.0.2 if Colab is used again. K16 SCREEN DONE (2026-10-08 13:54 KST, D-B-062): FAIL (seed-mean Dread −0.0025, consensus −0.0045 CI < 0); the Colab real-model line ends here unless a new single change is chosen.

### H-B-50 — TabPFN v2 nested stacker over same-fold rank OOF (CPU)
- Sources: D-B-046, D-B-003, D-B-037, D-B-035
- Hypothesis: A per-target TabPFN v2 classifier on family-wise rank OOF (CoAtNet5, RadR50 seeds, ResNet34, later VLM/DenseNet) plus 3–5 metadata fields, cross-fitted on the EXP099 folds, beats the equal rank mean by ≥ +0.003 on Dread and consensus.
- Evidence: the ridge stacker failed with a different fold lineage (D-B-003, −0.005) and per-target weights gave only +0.0013 (D-B-037); TabPFN v2 is the strongest small-tabular learner (Apache-2.0 + attribution) and our same-fold OOFs give the nested lineage it needs. DUPLICATE CHECK (02:15 KST): no plan item or discovery mentions DenseNet, BiomedCLIP, MRI-CORE, TabPFN or plane-aware TTA.
- Improvement: nested same-fold lineage and a small-data learner instead of a cross-lineage linear map
- Impact: 1
- Information: 2
- Confidence: 0
- Unblock: 0
- Diversity: 1
- Cost: 1
- Priority: 9
- Resource: CPU
- Parallel: YES
- Other: NONE
- Next test: needs the user's permission to pip-install tabpfn (v2, Apache-2.0 + attribution) and download its v2 weights; CPU cross-fit on the 4,343 OOF (rank inputs; consensus-pair fit as a variant); adopt only with ≥ +0.003 on Dread AND consensus and an unchanged result on a second seed of the base members; serving: rank-normalise per test cohort. PREPARED (02:32 KST, user permission): TabPFN v2 checkpoint storage/weights/prior-labs__tabpfn-v2-clf/tabpfn-v2-classifier.ckpt (29,016,968 B, sha256 f65a3568…, Prior Labs License = Apache-2.0 + attribution); isolated env .venv-fm (system-site-packages; tabpfn 2.2.1 + open_clip_torch 3.3.0 --no-deps; system timm/torch/numpy unchanged); Kaggle offline wheels storage/wheels/kaggle-py312/ (provenance.json). Use model_path= the v2 checkpoint explicitly (never the non-commercial 2.5/2.6 weights). PREMISE CHALLENGED (2026-10-08 11:10 KST, D-B-046 update): same-fold OOF is not nested for a learned stacker; run only as exploratory, or with inner OOF regenerated inside each outer training partition (needs base retraining). Not started.

### H-B-53 — Serve plane-aware flip TTA in the own arms (release v7) and test it on CoAtNet5
- Sources: D-B-056, D-B-048
- Hypothesis: Adding the plane-aware flip TTA to the RadR50 (and multi-head) serving arms keeps the +0.002 OOF gain at serving time within the 9 h budget, and the same TTA also helps CoAtNet5.
- Evidence: D-B-056: RadR50 Dread +0.0021 [+0.0011, +0.0031], consensus +0.0015, multi-head +0.0016, CI > 0; inference-only. DUPLICATE CHECK (11:14 KST): no plan item serves TTA; H-B-52 was the OOF test (done, removed).
- Improvement: from OOF evidence to the serving path, plus the untested CoAtNet family
- Impact: 1
- Information: 1
- Confidence: 2
- Unblock: 0
- Diversity: 0
- Cost: 1
- Priority: 9
- Resource: EITHER
- Parallel: YES
- Other: Kaggle (release run)
- Next test: (a) CPU: add a `--plane-flip-tta` option to a NEW release-v7 arm file (RadR50 centre3 arm), CPU parity on 24 local studies that the TTA output equals the local EXP124 TTA OOF for those studies, runtime +≈2,100 s/arm projected (D-B-048 constants); (b) GPU ≈15 min: EXP124-style TTA for the EXP099 CoAtNet5 fold checkpoints (needs the CoAtNet 3-slice loader) — adopt for CoAtNet only with ≥ +0.001 and CI > 0; ship TTA inside the next own-arm release (no separate submission).

### H-B-54 — RadR50 all-data final fit + late SWA (the 0.949 notebook recipe on our best family)
- Sources: D-B-058, D-A-050, D-B-049, H-B-33
- Hypothesis: A RadR50 trained on all 4,343 studies with SWA of late epochs (instead of five fold models) scores above the five-fold rad5 (0.925) on public.
- Evidence: the public 0.949 model and the 0.928 Raptor v5 both use all-data training + SWA of late epochs (D-B-058, D-A-050); our RadR50 is five ≈80% fold models with a fixed final epoch; swa16 (CoAtNet all-data SWA) is submitted tomorrow and reads the same lever on CoAtNet. DUPLICATE CHECK (12:53 KST): plan.md and discoveries.md searched for JEV, all-data/SWA on RadR50, test-time K on RadR50 and anatomy-specialist crops; only H-B-33 (CoAtNet all-data SWA, swa16) and D-B-032 (CoAtNet5 62 windows) are related and differ in family.
- Improvement: all-data + SWA on the RadR50 family (H-B-33 tests it on CoAtNet only); no OOF exists for an all-data model, so it is Tier A and public-judged
- Impact: 2
- Information: 2
- Confidence: 2
- Unblock: 1
- Diversity: 1
- Cost: 1
- Priority: 14
- Resource: GPU
- Parallel: YES
- Other: Kaggle (one release run)
- Next test: after the swa16 reading (registered: ≥ 0.923 supports, ≤ 0.914 against): build a RadR50 all-data runner (EXP113 recipe + the Tier-B bundle frozen by 10-12, SWA of 3 late epochs, seed 42; ≈ 40–60 min local GPU), serve it in place of the five folds with the same TTA, one public reading with a pre-registered forecast (≥ the rad5-bundle public); combine with the five-fold arm only if both are scored. GOLD58 IN FINAL FITS (2026-10-08 19:53 KST, user question 'should we also train on Gold like JEV did?'): proposal — Gold58 stays OUT of all development training (the only official-label yardstick; hidden-test labels are official reads like Gold58); the FINAL all-data AUG1 RadR50 fits (10-17–10-20) include the 58 Gold studies with their OFFICIAL labels at a weight registered before training (proposal ×5 vs report-label rows); no Gold58 evaluation after that; expected effect small (1.3 % of rows) but aligned with the test label definition. Decision to be cross-checked by fable + astra at the 23:00 review.

### H-B-55 — JEV-blended label source (riadmohamed42) as a fifth label head or blend for the RadR50 line
- Sources: D-B-058, D-B-041, D-B-051, D-B-013
- Hypothesis: Adding the public JEV report labels (as a fifth head of the multi-head RadR50, or blended into the Dread targets) improves the single model on public.
- Evidence: the public 0.949 model trains on labels blended with riadmohamed42's JEV labels (D-B-058); label work was the largest retained forum lever (+0.015 public, D-B-036); our multi-head with four sources gave +0.0048 consensus (D-B-051); the user asked about Jev on 10-08 (as a model it cannot read MRI; as a label source it is usable if public, since commercial LLM label extraction is allowed, D-B-013). DUPLICATE CHECK (12:53 KST): plan.md and discoveries.md searched for JEV, all-data/SWA on RadR50, test-time K on RadR50 and anatomy-specialist crops; only H-B-33 (CoAtNet all-data SWA, swa16) and D-B-032 (CoAtNet5 62 windows) are related and differ in family.
- Improvement: a fifth, differently produced label source inside the already-built multi-head machinery; a label change, so Tier A (public-judged)
- Impact: 2
- Information: 2
- Confidence: 1
- Unblock: 0
- Diversity: 2
- Cost: 1
- Priority: 13
- Resource: EITHER
- Parallel: YES
- Other: none
- Next test: CPU first: confirm riadmohamed42's JEV label set is a public Kaggle dataset with a usable licence (ask the user before downloading); measure it on Gold58 vs Dread / Steven v4 / Pilkwang v2 / relabel_v1 (never train on Gold58); if it is at least as accurate as the best source, add it as a fifth head (EXP117-style fold-0 screen, then five folds) and judge the bundle on public. UPDATE (2026-10-08 17:16 KST, user asked to use JEV): labels are local (storage/labels/riadmohamed42__jev-knee-labels, provenance.json); Gold58 macro AUC JEV HYBRID 0.9037 vs flight_v4 0.8991 / steven_v4 0.8927 / pilkwang_v2 0.8700 (JEV tuned on Gold58-containing 74 studies → exposed; D-B-068); next: fifth head on top of the AUG1 + multi-head bundle (EXP138) as its own one-change fold-0 screen, judged on consensus/flight/relabel (JEV-free references) and a public reading; JEV also blends steven + pilkwang (already heads) → expect partial redundancy. RULE (2026-10-08 19:53 KST): JEV-trained variants are never judged on Gold58 (JEV is Gold-tuned); judge on consensus-without-JEV / flight / relabel references and the ruler, plus one public reading. REVIEW (2026-10-08 23:10 KST, fable + astra): fable — JEV becomes H-B-63's label set, not a fifth head; astra — at most one two-seed fifth-head screen after EXP138, total loss scale preserved, no automatic five folds. Adopted: JEV enters through EXP143 first; the fifth-head screen is deferred (not dropped) behind EXP143 and EXP142.

### H-B-56 — RadR50 test-time window count sweep (K36 to K48 / K62), inference only
- Sources: D-B-058, D-B-012, D-B-032, D-B-056
- Hypothesis: Serving RadR50 with more windows (K48 or K62) raises its five-fold OOF by ≥ +0.001 at inference cost only.
- Evidence: K24 → K36 gave +0.0050 on RadR50 (EXP107) and +0.0034 on CoAtNet (D-B-012); the 0.949 model serves K94 on a 96-slice corpus (D-B-058); for CoAtNet5, 62 windows added only +0.0007 (D-B-032); RadR50 (centre slice) is untested. DUPLICATE CHECK (12:53 KST): plan.md and discoveries.md searched for JEV, all-data/SWA on RadR50, test-time K on RadR50 and anatomy-specialist crops; only H-B-33 (CoAtNet all-data SWA, swa16) and D-B-032 (CoAtNet5 62 windows) are related and differ in family.
- Improvement: the window-count lever on the RadR50 family, combined with the plane-aware TTA path
- Impact: 1
- Information: 1
- Confidence: 1
- Unblock: 0
- Diversity: 0
- Cost: 0
- Priority: 8
- Resource: GPU
- Parallel: YES
- Other: none
- Next test: extend the EXP124-style inference (new file) to K48/K62 for the EXP113 fold checkpoints (≈ 20–30 min GPU), Tier B (inference-only: one seed; gain on ≥ 2 of consensus/flight/relabel); serving-time cost counted in the bundle's ≤ 10% allowance. UPDATE (2026-10-09 09:22 KST, D-B-079): the 0.949 author lost 0.006 public going 94 → 24 windows on their CoAtNet; raises the value of an inference-only K sweep (K36 → K62/K94) for our RadR50 and the EXP143 model.

### H-B-57 — Anatomy-specialist recipe: fixed target-relevant planes and native-resolution crops for MCL / menisci / OA (higher-ceiling idea, low priority)
- Sources: D-A-014, D-A-026, astra direction review 2026-10-08 11:15
- Hypothesis: Per-target specialist inputs (fixed planes and native-resolution crops where the target lives) beat the whole-knee centre-slice MIL on the targets that carry most of the error.
- Evidence: astra (11:15 review) proposed it as the clearest untried mechanism; D-A-014 winning MRI recipes use localisation + sequence aggregation; D-A-026 concentrates error in MCL, menisci and OA; no local evidence yet. DUPLICATE CHECK (12:53 KST): plan.md and discoveries.md searched for JEV, all-data/SWA on RadR50, test-time K on RadR50 and anatomy-specialist crops; only H-B-33 (CoAtNet all-data SWA, swa16) and D-B-032 (CoAtNet5 62 windows) are related and differ in family.
- Improvement: a new input mechanism rather than another encoder swap
- Impact: 2
- Information: 2
- Confidence: 0
- Unblock: 0
- Diversity: 3
- Cost: 3
- Priority: 11
- Resource: GPU
- Parallel: NO
- Other: none
- Next test: only after the 10-12 milestone and if GPU time remains: a fold-0 two-seed screen of a coronal-only native-crop MCL/meniscus specialist vs the RadR50 control on those targets; no segmentation models from controlled-access datasets (external-data rule). UPDATE (2026-10-08 17:03 KST, astra review): fastMRI+ (and SKM-TEA) boxes give training-only LOCALIZATION supervision that the study-level mapping discards — candidate auxiliary attention/box loss after the per-dataset QA (H-B-62); absent boxes are not clinical negatives without demonstrated annotation coverage (keep native targets and unknown masks). REVIEW (2026-10-08 23:10 KST, fable + astra): astra's higher-ceiling idea — promote box-supervised localisation (fastMRI+ boxes give spatial supervision our pipeline discards; D-A-026: 47.6 % of the historical weak-label deficit sits in four targets) into a bounded specialist experiment, preserving unknown labels where annotation coverage is incomplete. Queued behind EXP143.

### H-B-58 — OAI-trained public 0.949 checkpoint line (user decision 2026-10-08): reproduce, then one pre-registered blend
- Sources: D-B-058, D-B-031, D-B-048, D-B-057
- Hypothesis: Our own submission of the public OAI-trained Raptor CoAtNet checkpoint reproduces ≈0.949 public, and one pre-registered rank blend with a diverse partner (the 0.943 anchor, or RadR50x5+TTA if the anchor does not fit the runtime) scores at least as high, giving a selection-B candidate in the medal region.
- Evidence: public notebooks with this checkpoint score 0.949 alone and 0.950 with a small ConvNeXt blend (D-B-058); the user decided to use it after the team-level disqualification risk was explained (agents.md USER EXCEPTION, 13:03); the host question was posted in topic 743416 (13:04). Decision consult (fable + astra): both put the as-is reproduction first and keep selection A = the non-OAI anchor; fable prefers 0.6 rank(0.949) + 0.4 rank(anchor) next (two near-equal diverse stacks; needs T(0.949) ≤ 4,200 s so anchor + 0.949 ≤ 30,600 s; wall-clock guard falls back to pure 0.949), with 0.85/0.15 + rad5 only if the anchor does not fit; astra prefers 0.9 rank(0.949) + 0.1 rank(RadR50x5+TTA) next and defers the anchor blend (runtime, unknown marginal value). DUPLICATE CHECK (13:08 KST): no plan item uses this checkpoint; H-B-32 blends the anchor with our own arms only.
- Improvement: a COMPLETE 0.949-tier candidate of our own plus at most one blend chosen by pre-registered weights, instead of copying the public notebook's tuned per-target weights
- Impact: 3
- Information: 2
- Confidence: 2
- Unblock: 1
- Diversity: 1
- Cost: 2
- Priority: 15
- Resource: EITHER
- Parallel: YES
- Other: Kaggle (release runs + submissions)
- Next test: DECISION (2026-10-08 13:08 KST): (1) 10-09 after the 09:00 KST reset: submit our private copy of the public 0.949 notebook unchanged (same checkpoint sha256 7e5315da…, K94, native-320 crop, mirror average) through tools/kaggle_submit_once.py — registered forecast 0.949 ± 0.002; STOP the branch for diagnosis if < 0.947 or ERROR (reported to the user, no retry); its submit-to-scored time is the runtime upper bound. (2) 10-10: ONE blend, weights fixed now and never tuned on public — if a measured T(0.949) leaves anchor + 0.949 ≤ 30,600 s: 0.6 rank(0.949) + 0.4 rank(anchor) per target (EXP126-style notebook: 0.949 stage first and written as the default submission, anchor stage under a wall-clock guard, blend only if both finish, never ERROR); otherwise 0.9 rank(0.949) + 0.1 rank(RadR50x5 + plane-aware TTA). Reading: keep the blend as selection B only if public ≥ (a) − 0.002 (display ties prefer the blend); otherwise B = (a). (3) 10-11: no further OAI variant; reassess. (4) Any host ruling against OAI → stop all OAI use at once and choose B from non-OAI COMPLETE candidates. Selection A stays the unchanged non-OAI anchor 56773226 (a hedge against predictive failure only; it cannot shield the team from disqualification). Prerequisites: Kaggle sign-in (quota read; pushes), the user's OK to pull the public notebook source (≈130 KB, nartaa/rsna-knee-0949-anatomical-mirror). CHANGE (2026-10-08 13:09 KST, user): no as-is reproduction SUBMISSION — step (1) becomes a Public3-only fidelity run (our private copy on the 3 visible test studies must match the public notebook's own output CSV, read in the browser; no slot); step (2) the single pre-registered blend is the only OAI submission. Provenance stays honest: the private notebook and the records name the public checkpoint and its authors (Apache 2.0 notices kept); B declined the user's request to omit OAI or present the checkpoint as our own model (rules §8.d deception risk, licence attribution). FAMILY CHECK (2026-10-08 13:15 KST, user asked to cover the other 0.949–0.950 notebooks): three are byte-identical copies of the single 0.949 model; the two 0.950 notebooks add the ConvNeXt 2.5D reader (competition data only, Apache 2.0, 0.929 alone, 0.944 at 30% with the community stack) with Gold58/public-tuned per-target weights. Candidates to register before any score: a non-OAI B option = anchor 0.7 + ConvNeXt reader 0.3 (the author's own reported 0.944 setting, uniform weight) and, inside the OAI line, the ConvNeXt reader as an alternative partner to RadR50 at a uniform weight (no per-target weights copied). REVISED (2026-10-08 13:22 KST, user direction 'no anchor blends; improve through the ruler'): no anchor variant. The OAI line is a side track: (1) Public3 fidelity run (no slot); (2) ONE light blend, weights registered now and never tuned — B1 = 0.85 rank(0.949) + 0.15 rank(ConvNeXt 2.5D reader, goodpjw2008, competition-data-only) or B2 = 0.9 rank(0.949) + 0.1 rank(RadR50x5 + plane-aware TTA); which one is submitted is decided after the fidelity run and a runtime check (decision consult). The main line is the ruler-judged RadR50 single model (H-B-34). ACTIVE AGAIN (2026-10-08 13:50 KST, user): use OAI actively; Public3 fidelity run now; B1/B2 light blends built (.local/exp128-oai-light); runtime check; every OAI-derived submission only with the user's explicit approval (first possible: 10-09 after 09:00 KST). Next ideas: the 0.949 model as a TEACHER for the RadR50 line (H-B-59). FIDELITY DONE (2026-10-08 13:53 KST, D-B-061): exact Public3 reproduction; B2 partner can use the two-seed RadR50 (D-B-060). GOLD58 (2026-10-08 15:18 KST, D-B-064/D-B-065): 0.949 0.9229, B1 0.9257, B2 0.9256, B3 0.9274 (+0.0045 [+0.0008, +0.0082]) → B3 is the OAI-line candidate (weights registered before results); astra builds the guarded B3 kernel (EXP133: 0.949 first as default, partners under measured-runtime guards, registered fallbacks B2/B1/pure 0.949); then a Public3 dry run; submission only with the user's explicit approval, one attempt. REVIEW (2026-10-08 17:03 KST): both reviewers — B3 is the right provisional selection B (submit only with the user's explicit approval; its own COMPLETE result, not Gold58 significance, decides); astra: measure B3 runtime on a larger sample before the release (EXP133 timing60 on Gold58 training studies, astra building); fable: ONE pre-registered successor by 10-14 = the same 0.8/0.1/0.1 weights with the AUG1 five-fold two-seed RadR50 swapped in (no re-weighting on public), COMPLETE by 10-18. B3-AUG1 (2026-10-08 19:07 KST): the pre-registered successor (same 0.8/0.1/0.1 weights, RadR50 partner = AUG1 2-seed×5 + TTA) is buildable now that the ten AUG1 checkpoints exist (sol building with the standalone release); judged on its own COMPLETE public reading, never re-weighted. PUBLIC FORECASTS REGISTERED (2026-10-08 19:21 KST, before any OAI-line submission; reading rules): B3 public 0.950–0.955 (0.949 public + Gold58 blend gain shrunk by ~half for the exposed-cohort optimism); ≤ 0.948 = the partners hurt on the hidden cohort → selection B falls back to the plain 0.949-class candidate only after a decision consult; B3-AUG1 public ≥ B3 − 0.002 (same weights, partner Δproxy +0.0004) — submitted at most one day after B3, never both as final selections. EXP140 standalone (non-OAI) forecast in H-B-34. USER APPROVAL (2026-10-08 19:51 KST): ONE OAI-line blend submission approved for 10-09 after 09:00 KST via tools/kaggle_submit_once.py — B3-AUG1 if its Public3 dry run passes and the fable/astra submission consult agrees, else B3; one attempt, no retry; a second OAI-line submission needs a new approval. REVIEW (2026-10-08 23:10 KST, fable + astra): fable — B3 is best-of-nine on exposed Gold58: expect ≈0.950, not 0.955; with OAI we sit in a ≈400-team cluster of 0.949 forks whose private order is noise plus partner strength → selection B should be pre-registered by structure (strongest independent partner set: 0.949 + our own near-equal model + 0.1 RadR50), never chosen by +0.001 public differences.

### H-B-59 — The OAI-trained 0.949 model as a teacher for our RadR50 (soft targets for silent / disputed cells, or an extra head)
- Sources: D-B-058, D-B-041, D-B-051, H-B-46
- Hypothesis: Training RadR50 with the 0.949 model's image-informed predictions as an additional target (fifth head, or soft fill of report-silent cells) raises our model's own strength and its value as a partner of the public SOTA.
- Evidence: the 0.949 model is the strongest image model available (public 0.949 vs our 0.925); D-B-041: the documented gains from noisy report labels come from filling report-silent cells with image-model predictions, more so the more model-weighted; our multi-head with four label sources gave +0.0048 consensus (D-B-051). Caveat: the teacher saw every training study (all-data), so its predictions on training studies are in-sample (smoothed labels, not OOF) — the student's Dread OOF cannot judge a label change (Tier A, public-judged). DUPLICATE CHECK (13:50 KST): H-B-46 is a teacher fill with OUR OOF teachers (held); no item uses the 0.949 model as a teacher.
- Improvement: a far stronger teacher than our own OOF, image-informed (and OAI-informed) signal transferred into a model we control and can serve cheaply
- Impact: 2
- Information: 2
- Confidence: 1
- Unblock: 0
- Diversity: 2
- Cost: 2
- Priority: 12
- Resource: GPU
- Parallel: NO
- Other: none
- Next test: (1) build the 96-slice native-384 corpus for the 4,343 training studies with the notebook's own reader (CPU, hours) and run the checkpoint on them locally at K94 + mirror (GPU, a few hours; overnight slot after the screens); (2) fold-0 screen: RadR50 with the teacher as a fifth head (EXP117 machinery) vs EXP117 F0 — judged on consensus/flight/relabel and, if promising, one public reading as part of a bundle. UPDATE (2026-10-08 17:03 KST, review — both reviewers' higher-ceiling idea): primary next test becomes DISTILLATION ON UNLABELED EXTERNAL STUDIES — the B3/0.949 teacher scores the ≈10k unlabeled fastMRI knee DICOM studies (never seen by the teacher → genuine out-of-sample pseudo-labels), RadR50 (AUG1 base) pre-trains or co-trains on them, then the usual competition training; competition OOF of the student stays clean only if the teacher's competition predictions are NOT used as targets (quarantine any OOF touched by an all-data teacher); judge on the ruler AND the partner proxy (a distilled student may lose diversity). Prerequisites: fastMRI DICOM GENERAL/PRODUCTION caches (EXP135), a teacher-inference kernel (0.949 runs on Kaggle only; local teacher = our own models or Kaggle batch jobs within the quota cap). Start by 10-10 if the bundle is on track. REVIEW (2026-10-08 23:10 KST, fable + astra): astra — lineage error: an all-data teacher (0.949/B3 saw every competition study) transmits validation-study information through external pseudo-labels, so a student's competition OOF is NOT clean even when the external images are unseen → ruler readings of such a student are optimistic; use outer-fold-excluded teachers or quarantine the evaluation (public-judged only). fable — pseudo-label the whole external pool (fastMRI DICOM ≈10k, OAI V00 2,400, MRNet/KneeMRI) with the B3 teacher run LOCALLY and feed H-B-63 (all-data fits need no OOF, so the leak is harmless there). Order: after the EXP143 pilot; first use = extra training data for H-B-63's second fit.

### H-B-60 — Borrowed-method screens on RadR50 (from slots A/C and public recipes): position-aware attention, late-epoch averaging/EMA, contextual slice transformer, backbone LR
- Sources: D-A-037, D-C-009, D-C-014, D-A-050, D-A-020, D-B-043, D-B-012, D-B-036, D-C-010, D-C-011, D-C-013
- Hypothesis: At least one of the borrowed mechanisms raises the RadR50 five-fold Dread by ≥ +0.001 (Tier B) with the gain on ≥ 2 other references and no loss of partner value.
- Evidence: decision consult (fable + astra, 13:53 KST) both rank within-series relative-position attention first (D-A-037 +0.00185 on CoAtNet fold 0, CI ∋ 0, Baker's down; D-C-009/014 goodpjw study transformer with relative positions, 0.929 reader); late-epoch averaging/EMA (D-A-050 v5 SWA of 3 late epochs; D-C-014 final EMA); a two-layer contextual block before MIL (D-C-014); backbone LR never tuned on RadR50 (D-A-020, D-B-043); fable also lists shifted-grid TTA (D-B-012 +0.0014/+0.0019, inference only → H-B-56), MRI-specific augmentation after AUG1, 288 px (Tier A) and fixed-physical-scale crop (D-C-010/011/013, 6/100 studies, tiny). DUPLICATE CHECK (13:53 KST): EXP127 covers AUG1/ADJ3/EP40 only; H-C-15 (slot C) is a full ConvNeXt study-transformer baseline, not a RadR50 head port.
- Improvement: ports of mechanisms other slots/teams found, each isolated on the RadR50 recipe with matched controls
- Impact: 2
- Information: 2
- Confidence: 1
- Unblock: 0
- Diversity: 2
- Cost: 2
- Priority: 12
- Resource: GPU
- Parallel: YES
- Other: none
- Next test: EXP129 (astra building): fold-0 seeds 42/43 vs EXP113 F0 / EXP118 F0 for POS (normalized within-series position into the MIL attention logits; constant-position control if the adapter changes), LATE (average of the last three epochs' weights re-inferred, paired against the same run's final epoch, plus EMA 0.998 arm), CTX (two-layer transformer over window embeddings before MIL), LRx0.5 / LRx2 (backbone LR); Tier A/B verdicts + the partner proxy (report only); queue after the EXP127 screens; then MRI-specific AUG2, shifted-grid TTA with H-B-56, physical-scale crop. RE-BASED (2026-10-08 17:03 KST, review): the un-augmented F0 screens are cancelled (both reviewers); EXP139 (astra) re-bases POS and LATE first, CTX second, on the AUG1 recipe with the existing EXP127_AUG1 F0 runs as controls; LRH/LRL dropped (low information); EP40 only as AUG1+EP40 later; ADJ3 rejected (D-B-067). REVIEW (2026-10-08 23:10 KST, fable + astra): bounded — finish LATE s42/s43 (running), CTX (EXP141 on Kaggle) and POS s42/s43 (queued after the bundle via the pin_memory shim); NO follow-ups (fable: ≤+0.002 standalone ≈ 0 in the blend). astra: EXP141 runs on T4 against LOCAL 4070 controls (no same-device pairing) → its delta is a screening signal only; a positive CTX must be confirmed locally before any bundle. UPDATE (2026-10-09 01:1x KST): EXP141 stopped for this week after three Kaggle failures (zip auto-extract, CUBLAS env, initial-state hash mismatch on the Kaggle torch/timm stack); CTX = optional lowest-priority LOCAL screen only.

### H-B-61 — External knee-MRI auxiliary training for RadR50 (MRNet ACL/meniscus/abnormal; KneeMRI ACL grade), training-only heads discarded at serving
- Sources: D-B-063, D-C-028, D-B-051, D-B-030
- Hypothesis: Pre-training or co-training the RadR50 trunk with auxiliary heads on MRNet (and KneeMRI) raises the five-fold Dread OOF on competition data by ≥ +0.004 (Tier A) — mainly ACL, MCL and menisci — at no serving cost.
- Evidence: MRNet labels ACL tears and meniscal tears on sagittal/coronal/axial knee MRI (1,370 exams), KneeMRI grades ACL on 917 sagittal volumes; our largest per-target weaknesses include ACL (RadR50 −0.011 vs CoAtNet5, D-B-030); training-only auxiliaries keep the validation on competition data, so the ruler applies directly (unlike all-data checkpoints); D-C-028 lists these as conditional training-only leads; user gray-area policy 14:36 KST. DUPLICATE CHECK (14:36 KST): no plan item in any slot trains on MRNet/KneeMRI; H-B-57 (anatomy crops) and H-B-59 (0.949 teacher) differ.
- Improvement: real image-level supervision for the weakest ligament/meniscus targets, judged by the existing ruler without a submission
- Impact: 2
- Information: 2
- Confidence: 1
- Unblock: 0
- Diversity: 3
- Cost: 2
- Priority: 13
- Resource: GPU
- Parallel: NO
- Other: none
- Next test: (1) data: the user registers at Stanford AIMI and downloads MRNet (≈6 GB) himself; KneeMRI download only with the user's permission (size stated first); store under storage/external_data/ with licence/provenance receipts; (2) CPU: convert to our centre-slice ×3 window format per plane at 336 px / 140 mm where spacing allows, map MRNet labels (abnormal, ACL, meniscus) and KneeMRI grades to auxiliary heads; (3) GPU: fold-0 two-seed screen — RadR50 trained on competition folds + MRNet auxiliary heads (loss weight fixed in advance, e.g. 0.5), vs EXP113 F0 / EXP118 F0; Tier A verdict on the four references + partner proxy; then five folds if it passes. UPDATE (2026-10-08 15:28 KST): KneeMRI local (736 vols); sol builds EXP132 (KMRI/MRN/BOTH variants, fixed 25% external steps, aux weight 0.5, bitwise-identical when disabled); MRNet zip still 0 bytes in the user's Chrome; fastMRI knee single-coil (1,172 coronal volumes) + fastMRI+ box labels arriving (user go) → add an FMRI variant (study-level labels from fastMRI+ boxes mapped to our targets: ACL←ACL sprain low/high, MCL←MCL sprain, menisci←Meniscus Tear/Displaced meniscal tissue (side from box x only if knee laterality is recoverable, else a shared 'any meniscus' aux head), OA←cartilage partial/full, Effusion←Joint Effusion, Baker's←Periarticular cysts (approx.), Contusion←Subchondral edema + Fracture/Contusion, Fracture←Fracture/Contusion; volumes with no boxes = negatives) after EXP132's build; unlabeled fastMRI knee DICOMs (≈10k studies) are a separate teacher pseudo-label idea (see H-B-59). REVIEW (2026-10-08 17:03 KST): one dataset at a time on the AUG1 base (EXP132-AUG1, sol building) after its H-B-62 QA; order KneeMRI → MRNet → fastMRI+ → OAI (bounded, separate branch) → SKM-TEA; the main RadR50 line stays OAI-free so a non-OAI line survives a late host ruling (fable); an OAI screen is judged on standalone ruler AND fixed-blend contribution (astra: OAI training is not redundant by design — measure it). RESULT (2026-10-08 21:54 KST): KneeMRI aux rejected in this configuration (D-B-070: seed-mean ΔDread −0.0027 [−0.0054, +0.0003], ACL itself −0.0027); MRNet s42 running, s43 next — if MRNet is also negative the shared mechanism (unaugmented external batches, weight 0.5) is the suspect, so fastMRI+/SKM-TEA would need a mechanism change (AUG1 on external batches, lower weight) before their screens; KneeMRI salvage (complete-tear label only + AUG1 externals) only on astra's advice. REVIEW (2026-10-08 23:10 KST, fable + astra): astra (salvage consult) — ONE bounded MRNet rescue arm (EXP142: BN in eval mode for external forwards, AUG1 on external windows from a separate RNG stream, weight 0.2, competition path unchanged; ≈1.5 GPU h, sol building), registered rule: seed-mean ΔDread > 0 with both seeds ≥ −0.001 → five-fold discussion, otherwise the auxiliary-head configuration is CLOSED; fable would skip it and fold the fix into distillation — adopted astra's bounded test because it runs in the lane-2 gap while EXP143 is being built. fastMRI+/SKM-TEA/OAI auxiliary screens stay paused. CLOSED (2026-10-09 02:27 KST, D-B-075): the EXP142 rescue arm was also negative (−0.0025 [−0.0040, −0.0008]); separate auxiliary heads are closed — external data continues via H-B-66 / H-B-59 / H-B-65 / H-B-37. Item kept only until the 05:00 review removes it.

### H-B-62 — Per-dataset external-data QA before training: visual/statistical cache check + zero-shot transfer AUC of our frozen RadR50 on each external PRODUCTION cache
- Sources: D-B-063, D-B-066, H-B-61
- Hypothesis: For each external source (KneeMRI, MRNet, fastMRI+ single-coil, OAI MOAKS, SKM-TEA), our frozen RadR50 two-seed ensemble scores the source's mapped labels clearly above chance (e.g. ACL AUC ≥ 0.75 on MRNet/KneeMRI, meniscus/effusion/OA ≥ 0.70 on OAI), which certifies that orientation, crop, intensity and label mapping are compatible before any training screen; a low AUC flags a preprocessing or mapping error to fix first.
- Evidence: user rule 16:38 KST (one dataset at a time; reasoned rejection or hold); EXP135 caches reproduce the competition contract byte-exactly from GENERAL but external volumes lack physical spacing (KneeMRI/MRNet) and laterality conventions differ; EXP136 mappings have known weak spots (periarticular cyst→Baker's, Hoffa→Synovitis, radiographic OA). DUPLICATE CHECK (16:38 KST): no plan item or discovery measures zero-shot transfer to external data; H-B-61 trains on external data, this item gates it.
- Improvement: turns 'no gain' into an interpretable verdict (data incompatible vs data uninformative) and catches preprocessing bugs before GPU training time is spent
- Impact: 1
- Information: 3
- Confidence: 2
- Unblock: 3
- Diversity: 1
- Cost: 1
- Priority: 16
- Resource: GPU
- Parallel: YES
- Other: none
- Next test: build tools/ext_zeroshot.py (sol) — K36 + plane-aware TTA inference of EXP113+EXP118 fold models on storage/cache/derived/EXT_<SRC>_RAPTOR64_336_C140 (fold models averaged, external studies are out-of-sample for every fold), per-target AUC with bootstrap CI vs storage/labels/external/<SRC>_12.csv and native labels (MRNet abnormal/ACL/meniscus, KneeMRI grade), plus a montage/statistics QA report next to competition studies; run per source as each PRODUCTION cache completes (MRNet and KneeMRI first); record each result as a discovery; only sources that pass enter H-B-61 training screens, one at a time.

### H-B-63 — A second 0.949-class model of our own: scale the EXP112 CoAtNet all-data + SWA trainer to the public 0.949 recipe (96-slice native-384, JEV-blend labels, K94), OAI-free first
- Sources: D-B-058, D-B-061, D-B-065, H-B-54
- Hypothesis: An independently trained near-equal model (public ≈0.94–0.95) blended at 40–50 % with the 0.949 checkpoint gains more (+0.004–0.006) than a weak partner at 10 % (+0.003 on Gold58), and an OAI-free version also strengthens the non-OAI selection A line.
- Evidence: fable review 17:03 KST (idea (b)); the 0.949 recipe is public and byte-reproduced by us (D-B-061: original384 full-field training, native-320 crop at inference, K94, mirror average, SWA of late epochs, JEV-blend labels — now local); swa16 (EXP112, all-data + SWA on our CoAtNet) is read publicly on 10-09 and is the go/no-go. DUPLICATE CHECK (17:03 KST): H-B-54 is RadR50 all-data + SWA (different family/recipe); no item trains the 0.949 recipe ourselves.
- Improvement: replaces copying a public checkpoint with an owned, near-equal partner whose lineage and data we control
- Impact: 3
- Information: 2
- Confidence: 1
- Unblock: 1
- Diversity: 2
- Cost: 3
- Priority: 14
- Resource: GPU
- Parallel: NO
- Other: none
- Next test: wait for swa16's public reading (10-09); if ≥ the registered forecast, port the 0.949 training recipe (public code) to the local queue with our corpus and JEV-blend labels, OAI-free, one seed, all-data + SWA (≈6–9 h local GPU) after the AUG1 bundle; judge on Gold58 direction + one public reading with registered forecast. REVIEW (2026-10-08 23:10 KST, fable + astra): both reviewers name H-B-63 the highest-ceiling change (a near-equal own partner at 40–50 % beside the 0.949; the RadR50 line at 0.1 weight converts +0.009 standalone into ≈+0.001 in the blend). swa16 is no longer a gate (fable: it tests one ingredient; astra: supporting evidence only) — astra builds EXP143 now (OAI-free, JEV-blend labels, Gold58 studies EXCLUDED from training so Gold58 stays a clean yardstick for it, CoAtNet public geometry, late SWA, one seed local, second seed on Kaggle after the 10-10 reset); GPU start when lane 1 frees after the EXP138 bundle; registered Gold58 blends 0.5/0.5 and 0.6/0.4 vs the 0.949 (not tuned); one public reading with a registered forecast. UPDATE (2026-10-09 09:22 KST, D-B-079): the 0.949 author published the training hyperparameters (16 epochs, backbone LR 4.01e-5, head LR 1.95e-3, WD 0.0588, 12 train windows, accumulation over 16 studies, 94-window/320-crop/mirror inference, Gold58-selected epoch average 14/11/12) — amend EXP143 PROTOCOL to these values with an a-priori epoch set before its first GPU run; OAI-free first, an OAI arm (masked PF OA / Lateral OA / Synovitis) only with the user's approval. UPDATE (2026-10-09 09:46 KST, Taeyang): the 9 h is our own registered LOCAL TRAINING cap, not a Kaggle rule — Kaggle's 9 h limit binds only the inference notebook (0.949-class inference measured ≈33 min per 1,300 studies). Taeyang lifted the cap for EXP143 v2: register a runaway guard (14 h ≈ 1.5× the 9.3 h projection) in Amendment A2 before launch instead of a 9 h stop; the frozen protocol-v1/S42 attempt stays as recorded (D-D-003). DELEGATED (Taeyang 09:46): a new Claude Code session takes this item over (same-host take-over) and runs: astra build report (B copies it to artifacts/experiments/EXP143_OWN0949/protocol-v2/BUILD_REPORT_astra.md) → CPU tests → prepare/validate v2 → probe (checkpointing on/off, only when the GPU is free) → full run after B's EXP149 students release the GPU (≈15:00), label = EXP149 winner if its screen is recorded before launch, else the default JEV HYBRID soft (registered before launch). Expected wall ≈9.3 h with checkpointing, ≈6.5–7 h without if memory allows (probe decides). RETURNED TO B (Taeyang, 2026-10-09 13:29): no other session took it over. astra build delivered 09:47 (27 tests unrun). Two routes: local 4070 Ti SUPER after the EXP149 screen, or the shared Vast RTX 3090 (24 GB, no gradient checkpointing needed, frees the local GPU for EXP149 advancement; needs ≈59 GB corpus upload, only ≈39 GB free on the 100 GB remote disk while C's paused files hold 61 GB → Taeyang decides on remote cleanup). EXP143 v2 is standalone (no paired control), so either GPU is valid under the cross-GPU rule; Kaggle T4 serving needs the parity check either way.

### H-B-64 — RadR50 with TRUE native ±1-neighbour 2.5D input (existing RAPTOR64N3 corpus) vs centre×3 — one bounded fold-0 screen
- Sources: D-B-067, D-B-028, D-B-066
- Hypothesis: Native ±1 slices as the three channels (not sparse cached slots) give RadR50 a fold-0 ΔDread ≥ +0.004 over centre×3, unlike ADJ3's −0.014.
- Evidence: astra salvage consult (17:17 KST): ADJ3's failure is most plausibly the multi-slice gap between cached slots (and/or channel behaviour), not 2.5D itself; RAPTOR64N3_336_C140_S02 exists with complete manifest and zero centre/mask mismatches; EXP108 (CoAtNet, native vs cached neighbours) −0.0011 CI ∋ 0 did not test RadR50 native vs centre×3; public ConvNeXt 2.5D reader 0.929 standalone (feasibility). DUPLICATE CHECK (17:17 KST): no item tests RadR50 on the N3 corpus.
- Improvement: the fair version of the rejected ADJ3 idea, at the cost of one fold-0 fit (corpus already built)
- Impact: 1
- Information: 2
- Confidence: 1
- Unblock: 0
- Diversity: 2
- Cost: 1
- Priority: 11
- Resource: GPU
- Parallel: NO
- Other: none
- Next test: AFTER the AUG1 five-fold, the AUG1+multi-head bundle and the first external screens: adapt the N3 loader to RadImageNet 2x−1 normalisation, one fold-0 seed-42 fit (native ±1) against the existing centre×3 and ADJ3 F0 runs with identical centres/labels/schedule/init; seed 43 only if promising; then test on the AUG1 base. Follow-ups only if positive: conv1 channel-averaged init; neighbours as extra grey windows at matched compute.

### H-B-65 — External expert-label yardstick: OAI MOAKS (10 targets) + fastMRI+ (4) + MRNet/KneeMRI (ACL) as a large secondary evaluation set for changes the five-fold ruler cannot judge
- Sources: D-B-069, D-B-070, D-B-071, D-B-042, D-B-065
- Hypothesis: Zero-shot macro AUC on the mapped external labels (OAI V00 MOAKS expert reads: ACL, menisci, compartment OA, PF OA, Effusion, Synovitis, Baker's; fastMRI+ boxes: ACL, MCL, Effusion, Baker's; MRNet/KneeMRI ACL) ranks NON-OAI candidate models in the same order as their public scores, so it can judge all-data fits (H-B-63/H-B-54), label-set changes and blends that the competition OOF ruler cannot, on thousands of studies instead of Gold58's 58.
- Evidence: label maps exist (storage/labels/external/*_12.csv: OAI 52,685 visit rows with e.g. Medial OA 34,282 labelled cells / 13,934 positive, Medial Meniscus 10,871 / 3,952, Effusion 7,485 / 1,222, Synovitis 6,858 / 660, Baker's 4,051 / 1,322; fastMRI+ 1,172 fully labelled on 4 targets); our frozen ensemble already transfers zero-shot (MRNet ACL 0.918, KneeMRI complete tear 0.918, D-B-069); the ruler is valid only within family (D-B-042) and Gold58 is small and exposed (D-B-065/068). DUPLICATE CHECK (2026-10-08 23:20 KST): H-B-62 uses zero-shot AUC once per dataset as a preprocessing gate, H-B-42 benches the competition ruler; no item uses external expert labels as a standing yardstick for model choice.
- Improvement: turns the downloaded external data into an evaluation asset independent of report-derived labels; the auxiliary-training failure (D-B-070/071) does not affect this use
- Impact: 2
- Information: 3
- Confidence: 2
- Unblock: 2
- Diversity: 2
- Cost: 1
- Priority: 18
- Resource: GPU
- Parallel: YES
- Other: none
- Next test: (1) validation first (H-B-42 style): score the scored non-OAI models we have checkpoints for (RadR50x5 public 0.925, CoAtNet5 0.914, CoAtNet5+RadR50 0.926, swa16 and EXP140 after their readings) on the OAI V00 + fastMRI+ PRODUCTION caches (inference only, ≈20–40 GPU min per model) and check the rank agreement with their public scores; (2) only if the order agrees, register it as a secondary yardstick for H-B-63/H-B-54/H-B-38 decisions. Never for the 0.949/B3 line (OAI-trained → contaminated on OAI). Domain shift (OAI 3T, DESS/IW-TSE) limits it to direction, not size.

### H-B-66 — Joint training with SHARED heads: external studies with expert labels mapped onto our 12 targets, mixed into the same batches (masked loss for unlabelled targets), AUG1 for all, external-safe BatchNorm — the way the 0.949 used OAI
- Sources: D-B-058, D-B-070, D-B-071, H-B-61
- Hypothesis: External data helps when it supervises the SAME 12 outputs (labels mapped, missing targets masked) inside mixed batches with the same augmentation and BN statistics that are not shifted by external images, unlike EXP132's separate auxiliary heads on native labels (ACL-dominated) in separate unaugmented batches (D-B-070/071: −0.0027/−0.0035).
- Evidence: the 0.949 checkpoint was trained on competition + OAI data (D-B-058) and leads every non-OAI recipe; OAI MOAKS maps 10 of 12 targets with many expert-labelled cells (storage/labels/external/OAI_12.csv), fastMRI+ maps 4 targets on 1,172 studies, MRNet/KneeMRI map ACL only; our failures share the separate-head/separate-batch/train-mode-BN mechanism. DUPLICATE CHECK (2026-10-08 23:25 KST): H-B-61/EXP132/EXP142 use separate auxiliary heads; H-B-63 is an all-data CoAtNet fit (OAI-free first); no item tests shared-head mapped-label joint training on the fold-0 ruler.
- Improvement: changes the supervision target (shared, mapped heads), the batch composition (mixed) and BN handling at once — the 0.949 recipe's way of using external data — judged on the clean fold-0 ruler (expert labels, no teacher, no leakage)
- Impact: 3
- Information: 2
- Confidence: 1
- Unblock: 2
- Diversity: 2
- Cost: 2
- Priority: 16
- Resource: GPU
- Parallel: NO
- Other: none
- Next test: after EXP142 reads out: (1) non-OAI arm first — fastMRI+ single-coil (coronal PD/PD-FS → coronal slots; ACL/MCL/Effusion/Baker's mapped, other targets masked) mixed ≈1 external per 4 competition studies in the same batches, AUG1 on all, domain-specific BN statistics (or BN eval for external samples), last 3 epochs competition-only, fold-0 seeds 42/43 vs EXP127 AUG1 controls; (2) OAI arm (OAI V00 MOAKS, 10 targets; OAI-derived → local experiment fine, any submission needs the user's approval). Registered rule as Tier A/B; one dataset at a time. UPDATE (2026-10-09 09:22 KST, D-B-079): the 0.949 recipe used exactly this mechanism — 2,399 OAI baseline knees (= our V00 cache), shared heads, ONLY PF OA / Lateral OA / Synovitis supervised from OAI, other cells masked, RSNA-based pos weights → 0.940 → 0.945 (+0.003 more with the 384 full-field recipe). Copy that design for the OAI arm; extra MOAKS targets as a separate extension.

### H-B-67 — Vast RTX 3090 parity before use: inference parity with local (as D-B-008 did for T4) plus a training-equivalence re-run of a known local fold
- Sources: D-B-008, D-A-025, D-A-045, D-C-054
- Hypothesis: With the same pinned stack (torch 2.5.1+cu121, timm 1.0.19) and the parity policy, the Vast 3090 reproduces local inference within the T4 tolerances (fp16 max 2e-3, AUC |Δ| ≤ 1e-4), and a re-run of EXP127 AUG1 fold 0 seed 42 on the 3090 lands within the local seed spread (|S42 − S43| on fold 0) on Dread and consensus, so Vast-trained models are interchangeable members (still never paired against local controls).
- Evidence: D-B-008 — T4 inference max 5.5e-4 with the pinned stack, but Windows-vs-Linux training inputs/init differ, so trajectories cannot be bit-matched; D-C-054 shows the 3090 runs our recipes stably (two 8-epoch fits, zero skips). before any B use of Vast. DUPLICATE CHECK (13:33): H-B-16 (T4 parity) is closed in D-B-008; no plan item covers the 3090; C's D-C-038 is a determinism diagnostic of a different recipe.
- Improvement: adds a TRAINING-equivalence test that the T4 probe never had (D-B-008 stopped at "do not pair"), and covers the RadR50 AUG1 + flip-TTA serving path and a CoAtNet attention path.
- Impact: 2
- Information: 3
- Confidence: 2
- Unblock: 3
- Diversity: 1
- Cost: 1
- Priority: 19
- Resource: GPU
- Parallel: YES
- Other: Vast 54941539
- Next test: astra builds tools/vast_parity_3090.py + PROTOCOL (dispatched 13:33); B runs local-infer when the local GPU is free, starts Vast 54941539 (remap risk D-C-049), uploads the staged package + fold-0 corpus (≈30 GB max, fits the ≈39 GB free disk without deleting C's files), runs remote_run.sh, compares. PASS → EXP143 v2 may train on Vast; FAIL → Vast only for complete same-GPU screens.

## Agent: C

### H-C-22 — HOLD (user): Bind funded Runpod RTX3090 deployment after completed offline preparation
- Sources: Taeyang direct request2026-10-08 19:25KST; D-C-031; D-C-023, D-C-026, D-C-030; existing H-C-17/H-C-06
- Hypothesis: A live quote, direct SSH and exact remote environment/input binding can admit the already prepared compact H-C-17 on one Community on-demand3090 without repeating Vast failures or science.
- Evidence: (updated 2026-10-08 20:08 KST) D-C-031 offline preparation remains complete and unchanged. User completed Google login/funding; live balance USD10, autopay Disabled. User-approved SSH public key registered and fingerprint matched; .local/runpod-3090-prep-20261008/live-readiness-ssh.json and ssh-key-registered.jpg. D-C-023 records current direct-SSH capacity limitation. No Pod/endpoint, remote test or admission exists; no API-key/OAuth account tooling authentication. Referral bonus is unconfirmed and does not block use of the funded balance.
- Improvement: Remaining phase is actual provider binding, not more duplicate offline preparation. Preserve exact reviewed artifacts; use versioned repair only for newly diagnosed mismatch. C17 sender retains partials and refuses automatic retry; future largeC06 resume remains separate.
- Impact: 2
- Information: 3
- Confidence: 2
- Unblock: 3
- Diversity: 0
- Cost: 1
- Priority: 17
- Resource: CPU
- Parallel: YES
- Other: NONE
- Next test: (updated 2026-10-08 20:15 KST) USER HOLD: user stopped Runpod and redirected retry to Vast/H-C-23. Preserve preparation; no Runpod deployment until resumed. Historical next gates: Login/payment/key registration completed; inspect liveoneCommunityon-demandRTX3090 quote/stock/balance,32GBRAM/12vCPU/directTCP22,20GBcontainer+80GBpersistent and registeredpublickey BEFORE singlecreate. Offlinev2requestcannotitselfguaranteeCommunitypublicIP; consoleorverifiedequivalentmustreconcileallconstraints. Firstsetup/transfer/C17/export proposedcap1.50USDmustbefundedandregistered, noautotopup/extraGPU/Securefallback. Active supervisionorfunctioningboundedcoststopbeforeunattendedpaiduse; pausedrsna/rsna-6staypaused. FreshversionedauthenticatedSSHtrust, setup1200s+30skillgrace, eachtransfer600s/max1attempt, exactimmutable staging and realprocess/reservation/source/environment/budget evidence; newindependentbindingreview and<=600sfreshadmissionwith>=120srunway. Claim H-C-17 onlyaftergates; launchreviewedvenvPATH->gpu_queue->outer900s/noretry. NoVastrestart,local/Kaggle/ColabCodexGPU oroldscientificdispatcher. C06same3090pair7200sandlargeversionedtransportremainH-C-06,neverlocalpairedcontrol. Local readiness phase is complete; do not repeat it while waiting for compatible stock. Basic proxy SSH is documented but cannot be substituted into the reviewed direct transfer helper. Any alternate transport needs a versioned evidence/review supplement before deployment; no blind rental or Secure/3090Ti price substitution.

### H-C-15 — Prepare a controlled standalone six-slot ConvNeXt study-transformer baseline
- Sources: D-C-022, D-C-021, D-B-049, D-C-020, D-C-015, D-C-014, D-C-010, D-A-044, D-A-050; goodpjw2008/rsna-knee-2-5d-convnext-reader v1
- Hypothesis: The fully shared six-slot ConvNeXt/position/study-transformer/final-EMA recipe can become a controllable alternative image baseline within a bounded one-fold budget, and can test a materially different recipe from our prior four-slot ConvNeXt-S.
- Evidence: D-C-022(PENDING cross-host) completes one exact-reviewed V3 CPU integration sample; reuse its receipts. D-C-021 V1 failure was preserved and addressed by a materially revised new-path ancestry contract, not replay. Whole-cache/serving/GPU readiness remains unadmitted. D-C-021(PENDING) records new16-test CLI and one FAILED worker-parent admission before decode; no native/backend success. Original attempt and CLI frozen; no replay. V2 Windows redirector repair and four real/synthetic launcher tests are complete at .local/hc15-launcher-repair-v1/; actual sample dispatch stays disabled. D-B-049 VERIFIED now supplies the registered RadR50 standalone public result; this removes its availability wait, but does not establish ConvNeXt benefit or admit cache/GPU expansion. D-C-020(PENDING cross-host) closes C05 invalid. D-C-015(VERIFIED) records frozen inputs/sample success and13+16 tests passed in an isolated CPU-only environment; system CUDA-build CPU operations still crash. The input-selection/checking defects are repaired in V2; full-cache/pilot/admission gates remain and execution is hard-disabled. Latest prototype is NOT_EXECUTION_READY. The20:01 code-only phase added independently reviewed storage/decoder modules and passed29 new synthetic tests plus2 process-supervision tests; D-C-015 is cross-host VERIFIED for CPU scope only. Reuse .local/hc15-cache-supervision-v1/result.json; real allocation/CLI/backend and serving remain unadmitted. D-C-014(VERIFIED by Claude Code (B) at16:48) binds full public source and accessible dependencies, not reproduced efficacy. Source author .929 is three folds and ~3h/fold has unknown hardware; T3455 requires6034updates and the separate cache upper bound is214.20GiB. Existing current B control binds G1 and Dread labels. Prior own ConvNeXt-S public .909 is not a causal comparator for the author's .929. (updated 2026-10-08 15:47 KST) D-C-022 now binds the complete capped23992series count inventory and parent arithmetic/order/hash review. Conditional pixel upper bound107895914496bytes (100.486GiB) replaces the looser worst-case for storage screening only; current100GB residency remains unadmitted. No efficacy/feasibility evidence changes Priority15.
- Improvement: Port one complete standalone recipe, not another geometry fragment or extra label blend. Reuse G1 and original Dread training/evaluation to avoid overlap with B27/28/29, explicitly recording these deviations from author v0. One whole visual/optimization package is the variable; no individual-mechanism attribution.
- Impact: 2
- Information: 3
- Confidence: 0
- Unblock: 1
- Diversity: 2
- Cost: 1
- Priority: 15
- Resource: CPU
- Parallel: YES
- Other: NONE
- Next test: (updated 2026-10-08 15:47 KST; D-C-022 capacity evidence) Metadata phase is COMPLETE; reuse capped-counts-v1.csv and parent-result-review.json, never repeat the inventory/GROUP BY or old sample. Conditional pixel bound107895914496bytes plus overhead cannot currently admit all-series uncompressed residency on100GB.  (updated 2026-10-08 16:52 KST) Storage-design proposal complete at .local/hc15-vast-capacity-20261008/storage-proposal-v1.md; reuse it. Compressed resident all-series storage is the first candidate for a measured screen, with exact original-NPY byte hashes; network-LRU remains conditional because access trace, bandwidth and repeated cost are unmeasured. The separately registered tiny lossless probe is now complete (D-C-022); reuse its exact-byte receipt, no repeat. The one-study result is not a population ratio. Next register a bounded production-cache/serving admission proposal with exact disk/transport/read-path limits before any larger compression/decode phase. Register any new CPU implementation/decoding phase separately before execution; no automatic budget renewal or remote transfer from this note. (updated 2026-10-08 15:10 KST, user routing instruction) Any future GPU pilot/fit/evaluation is Vast-only. Full-cache fit within the current100GB disk is unproven; register storage/streaming and transfer budgets before expansion, without adding disk or rental automatically. Local CPU preparation requires spare headroom. The4h limit remains a conditional cap, not a3090 estimate; old local B controls are descriptive references only unless matched remote control validity is established. Reuse completed CPU contracts/tests and D-C-022; do not rebuild the producer/supervisor or repeat the first-study sample. Highest next CPU step is a bounded production-cache/serving admission proposal: exact full-cohort/all-series source+index bindings, resumable aggregate time/byte/disk-reserve budgets, allowed environment, and wrapper/evidence validator. Register its separate phase scope/cost before implementation or large decoding; neither completed45min/15min phase is silently renewed. Historical worst-case pixel bound226,416,918,528bytes is now narrowed to conditional107,895,914,496bytes by the complete metadata inventory; one-study timing is not full-throughput evidence. A valid subset projection must preserve series order, retained counts,6slots,384canonical->256views and source membership. Scientific prospective recipe stays G1 T3455/V888,12targets, Dread train/eval/mask/metric, Gold/quarantine excluded, generic initialization, k8train/k12eval, transformer/plainBCE,14epochs/6034updates/effective batch8/finalEMA; microbatch1 accumulation8 and epoch-exact resume, initial zero loader workers. Require source/init/environment hashes, no UID/label holes, every study has a valid series. Keep full cache/GPU disabled until exact independent review and concrete total budget/serving evidence pass; sample success is not model readiness. Any later GPU pilot is<=10min, projected full one-fold incl eval<=4GPUh, with fixed primary B comparator/practical decision rule registered before fit; only existing reservations on the target Vast instance precede admission. No automatic extra sample, fold, submission or hidden recipe adjustment on failure. (updated 2026-10-08 18:29 KST; Taeyang exitsVast) GPU phases BLOCKED_EXECUTOR: priorVast resumption is withdrawn, no alternatepaid/local/Kaggle/ColabC GPU approved. Preserve all prepared/frozen artifacts; CPU-only work may proceed with spareheadroom. No repeated preparation/fit to fill resources. (updated 2026-10-09 04:49 KST; D-C-046) Bounded production-cache proposal/validator phase COMPLETE; reuse .local/hc15-production-admission-v1/. Next CPU prerequisite is a separately registered versioned compressed-only serving loader and exact-original-NPY/RNG/read-path parity. Future build/transport limits are proposals only; no large decode/full-cache/GPU action admitted. Do not repeat metadata/probe; independent full production admission remains unresolved.

### H-C-21 — Deferred training-only plane/contrast auxiliary supervision on RadR50
- Sources: D-C-029; D-A-034, D-A-037, D-B-049, D-B-050; aneurysm second-place writeup/code revision1ead0ba4dc00bcd84591269995dde6a8446ca3ad
- Hypothesis: A small training-only slot-classification loss on the shared slice features improves pathology ranking without an extra inference head or extra serving windows.
- Evidence: D-C-029 PENDING cross-host identifies source code for a modality CE head and author-reported training stability, with no isolated AUC ablation. Our five SLOTS64 categories already describe plane/contrast. D-A-034's frozen-feature slot bias, B60 POS/CTX/LATE/LR, B47 report-label heads and B61 external-data pathology heads do not test this encoder auxiliary gradient. Existing broad RadR50 owner B34/B60 must be checked again before implementation; if equivalent auxiliary work appears, reuse it and retire this item.
- Improvement: Use existing competition-train images and existing slot metadata only, no new annotations/data/cache; classify five slot categories from pre-MIL image features with Linear(F,5), mask missing/sentinel windows, BCE +0.05*mean(valid CE), and discard auxiliary head at serving. The0.05 scale is prospectively fixed, not a proven optimum. No plane/slot identity is fed into the auxiliary classifier.
- Impact: 1
- Information: 2
- Confidence: 1
- Unblock: 0
- Diversity: 2
- Cost: 2
- Priority: 10
- Resource: EITHER
- Parallel: NO
- Other: NONE
- Next test: (updated 2026-10-08 15:10 KST, user routing instruction) All fits/evaluations are on the same Vast RTX3090; do not reuse local B runs as a matched remote control. Default remote scope is lambda0/.05 x seeds42/43 (four fits), unless exact same-Vast control reuse is independently validated. Those controls must fit the existing4h aggregate cap; otherwise remain deferred and prospectively revise budget before any fit. CPU preparation is allowed locally only with spare headroom. Deferred behind current higher-priority work and all active reservations on the target Vast GPU; no automatic fit from this plan. Bounded CPU preparation<=20min registers fold0 seeds42/43, K12training/K36evaluation,20epochs, unchanged source/init/labels/split/mask/metric/environment and actual/effective batch. Prove added-head initialization cannot alter sampling/dropout/backbone RNG; instantiate it in both lambda0/.05 arms or reuse controls only with deterministic equivalence. Independently review exact source/contract. Feasibility<=5min within total<=4GPUh for up to4same-machine fit/evaluation arms; projected overshoot stops before full fits and requires new budget evidence. Report matched seed/target deltas across four correlated references and fixed partner proxy; sign replication decides whether full5fold confirmation is worth further cost, not onefold adoption. Full adoption uses current frozen Tier-B safeguards, no threshold change or automatic5fold/seed/submission. If the easy auxiliary task is solved without pathology gains, do not expand to a loss-weight sweep. Coordinate result integration with B's main family instead of launching duplicate ownership. (updated 2026-10-08 18:29 KST; Taeyang exitsVast) GPU phases BLOCKED_EXECUTOR: priorVast resumption is withdrawn, no alternatepaid/local/Kaggle/ColabC GPU approved. Preserve all prepared/frozen artifacts; CPU-only work may proceed with spareheadroom. No repeated preparation/fit to fill resources.

### H-C-20 — RadR50 training K12 to K36 with unchanged K36 inference
- Sources: Taeyang direct request2026-10-08, amended2026-10-08 14:26 KST from training K32 to K36 while keeping inference K36; D-B-062, D-B-012; H-B-34 main-family ownership and H-B-56 inference-only distinction
- Hypothesis: Training the same RadR50 on K36 rather than K12 windows can improve bag coverage/learning even though the smaller K16 change failed, while serving remains the unchanged K36 single-model recipe.
- Evidence: D-B-062 VERIFIED within its corrected F0-only scope reports fold0 two-seed K16-minus-K12 Dread -0.0025 and consensus -0.0045 under its exact protocol. That is negative evidence for K16, not a measured monotonic response or a result for K36. No exact K36 training experiment/heading/Hypothesis was found in discoveries or the active queue. H-B-34 is the broader single-family line, H-B-56 changes inference windows only; coordinate implementation with B and retire this item if an equivalent K36 screen is already registered before activation. D-B-012's older CoAtNet inference response is not evidence of RadR50 K36 training efficacy. Confidence remains0; benefit and runtime are unmeasured. Corrected observed K16 runs took85.32/61.80min, so the4GPUh K36 cap must be checked by measured feasibility and cannot be assumed to fit.
- Improvement: A materially larger TRAINING-bag contrast instead of an unchanged K16 retry; freeze evaluation at K36 and preserve the original K16 failed verdict. Fixed study/update/epoch budget means extra training-window compute, not a compute-matched efficiency claim. Distinguish windows, unique centre slices and stored corpus density; K36 need not mean36 distinct informative slices in every study.
- Impact: 1
- Information: 2
- Confidence: 0
- Unblock: 0
- Diversity: 1
- Cost: 3
- Priority: 7
- Resource: GPU
- Parallel: NO
- Other: NONE
- Next test: (updated 2026-10-09 04:54 KST; D-C-048) OriginallocalK36/batch8 pilot FAILED after34loggedupdates with driverreset; fullpairedfits/screen absent. No unchangedretry, silentmicrobatch/epoch/precisionchange or newGPUjob. Reuse exactsource/control/coverageCPUreceipts and preserve allfrozenoutputs. CPU-only revisedpairedmemory proposal is ready at .local/hc20-local-k36-v1/revised-paired-memory-proposal.md; NOT_EXECUTION_READY. Proposed checkpointpolicy must apply tobothK12/K36 with unchangedactual/effectivebatch8,fold0,seeds42/43,20epochs/8620updates,labels/metric/init/optimizer/schedule; explicitlyverify BNrecomputebuffer/RNGbehavior. Assumefreshsame-localK12controls (4fits) unless strict equivalence proven. Newbounded<=300s measuredfeasibility plus allfits/evals must fit remaining14228.781s oforiginal4h cap, or register explicitrevisedtotalbudget BEFORE fitting. Parentonlyperuser, no consultants/sub-agents, no5fold/submission. H-C-22remainsHOLD; otherGPUroutesnotamended.

### H-C-07 — T-only MRI 자기지도 적응의 타당성 관문 및 동일 모델 대조 (from H-A-07)
- Sources: D-A-039, D-A-014, D-A-037, D-A-046, H-C-06
- Hypothesis: 영상에 맞춘 자기지도 적응은 같은 encoder/pooling의 적응 없는 모델보다 후속 고정 supervised8epoch 성능을 개선할 수 있다.
- Evidence: VERIFIED D-A-039의 generic DINO 열세는 관측되었으나 원인이 domain mismatch라는 증거는 없다. D-A-037의 대조-B +0.0003은 residual attention을 넣고 optimizer를 재시작한 특정 추가 지도학습 설정으로, SSL 실패나 보편적 긴 학습 실패를 뜻하지 않는다. 완료된 SSL 구현/실측/누출 없는 lineage 계약은 없다. 다른 대회 사례(D-A-014)는 전이 효능의 직접 증거가 아니다. 따라서 Confidence1->0; 현 단계는 조건부 대안이다.
- Improvement: 기존 'SSL후 raw-B'만으로 원인을 해석하지 않고, 선택된 동일 DINO encoder/pooling/초기값의 no-SSL arm을 primary 대조로 둔다. B는 별도 성능 참고다. SSL 단계에서 T와 그 내부 train-only 설계만 사용하고 V/Gold/연결 group 및 그들을 본 teacher/checkpoint를 제외한다. 사전학습 토큰/초기값 계보를 모르면 불확실성으로 남긴다. 새 무라벨 노출량이 추가되므로 label-free 적응 package 효과이지 compute-matched 감독 대비 우위가 아님을 명시한다.
- Impact: 1
- Information: 1
- Confidence: 0
- Unblock: 0
- Diversity: 3
- Cost: 3
- Priority: 7
- Resource: GPU
- Parallel: NO
- Other: NONE
- Next test: (updated 2026-10-08 15:10 KST, user routing instruction) Any future SSL, no-SSL control, supervised fitting and evaluation use the same Vast RTX3090. Local CPU design only with spare headroom. HOLD and requirement to register full control-inclusive budget remain; no local/Kaggle/Colab fallback. HOLD, 자동 GPU dispatch 금지. H-C-06 완료만으로 해제하지 않는다. 그 결과와 새 domain/representation 근거가 SSL 비용을 정당화할 때에만 CPU 설계<=3h: 단일 masked-image objective/손실영역/증강/학습량/초기값/no-SSL control 재사용/누출 검사/성공 update/metric/runtime 계약을 구체화하고 plan-change 이벤트로 관문 해제를 기록한다. 재사용 불가 시 두 arm 비용을 먼저 재산정하며 묵시적 추가 fit 금지. 그 계약에서 SSL<=2h와 supervised8epoch1fit 및 pilot/eval 포함 총 cap을 확정하기 전에는 launch하지 않는다. 최종 동일 V888 원 Dread로 adapted-noSSL primary, adapted-B secondary와 paired12소견 CI를 보고하되 clinical/private 효과로 해석하지 않는다. 해상도/LR/pooling 동시 탐색과 자동5fold/제출 없음. (updated 2026-10-08 18:29 KST; Taeyang exitsVast) GPU phases BLOCKED_EXECUTOR: priorVast resumption is withdrawn, no alternatepaid/local/Kaggle/ColabC GPU approved. Preserve all prepared/frozen artifacts; CPU-only work may proceed with spareheadroom. No repeated preparation/fit to fill resources.

### H-C-08 — 384/12epoch 전체 레시피의 추가 fold 재현, 신규 근거 전 보류 (from H-A-03)
- Sources: D-A-042, D-A-004, D-A-046, D-A-050; D-B-020 (VERIFIED after scope correction, 역사적 제출 기록만 참조)
- Hypothesis: EXP088의 전체384/12epoch 레시피가 다른 G 분할에서도 같은 분할/seed의224 B레시피보다 높은 성능을 내어 배포 후보 선택을 바꿀 수 있다.
- Evidence: VERIFIED D-A-042: 노출 weak V888 한 seed에서 +0.001958, CI0포함, MCL -0.017214, 1.5T -0.002712 / 3T +0.008758, 학습12343초 versus2610초. D-A-050의 완료 레시피 감사는 이 실험의 조건이던 H-A-14를 닫았으나 추가384 fold를 우선할 새 효능 근거를 제공하지 않았다. 공개 v5의 정확한 train/label 계약은 unknown이다. D-B-020은 일반 dev-public 전이 주장을 철회한 한정된 역사적 기록으로 VERIFIED이며, 일반화 근거로 사용하지 않는다. Confidence1->0은 투자 확신 하향이지 관측된 작은 양의 효과의 삭제가 아니다.
- Improvement: 해상도뿐 아니라 epoch/precision/microbatch 등이 바뀌는 전체 레시피 재현이라고 명시한다. 신규 fold에서 두 arm의 study/split/seed/labels/mask/metric/평가K는 고정하되 각 레시피의 나머지 차이를 나열한다. 순수 해상도 효과를 묻는다면 다른 matched 계약을 사전등록해야 하며 이 실험의 해석으로 대체하지 않는다.
- Impact: 2
- Information: 1
- Confidence: 0
- Unblock: 0
- Diversity: 0
- Cost: 3
- Priority: 6
- Resource: GPU
- Parallel: NO
- Other: NONE
- Next test: (updated 2026-10-08 15:10 KST, user routing instruction) Any future384/224 paired fits and evaluation use the same Vast RTX3090. Historical4.15h local timing is reference only; remote feasibility and control-inclusive cap must be registered before HOLD release. CPU preparation locally only with spare headroom. HOLD, 자동 GPU dispatch 금지. 'H-A-14 완료'는 실행 조건으로 더 이상 사용하지 않는다. 추가 fold의 결과가 바꿀 명시적 후보 결정과 새 efficacy/runtime 근거가 생기면 재심사한다. 재개 시 G2 384+224 same-split/seed pair를 새 경로에 등록하고 기존 시간 기준 약4.15GPUh(12343+2610초)에 준비/eval 포함 hard cap을 먼저 확정한다. 추가 앙상블이라면9h 전체 추론 예산 적합성도 필요하다. 고정12소견/group/조건부 CI를 보고, 작은 차이를 resolution-only 인과나 public/private 보장으로 확대하지 않는다. 자동5fold 금지. (updated 2026-10-08 18:29 KST; Taeyang exitsVast) GPU phases BLOCKED_EXECUTOR: priorVast resumption is withdrawn, no alternatepaid/local/Kaggle/ColabC GPU approved. Preserve all prepared/frozen artifacts; CPU-only work may proceed with spareheadroom. No repeated preparation/fit to fill resources.

