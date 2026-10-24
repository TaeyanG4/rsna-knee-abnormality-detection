<!-- Investigator report for postmortem-20261025.md (B, 2026-10-25). Read-only review of discoveries.md, handoff archives, plans and memory; citations are file:line or D-IDs as found. -->

## 1. Own model families

| Family (date) | Inputs / geometry | Best result |
|---|---|---|
| EXP045 ConvNeXt-S, 5 folds (09-13) | 288 px | 0.909 / 0.891 (D-A-044) |
| B224, R384 single split; CoAtNet5 (10-07) | Raptor 64-slot cache, 140 mm centre crop, train on 12 windows, serve on 24–36, ≈80 % of data per model | 0.910 / 0.913 / 0.914 (D-B-020, D-B-027) |
| RadR50 ×5 (10-08); AUG1 two seeds (10-09) | centre slice ×3, 224 px, 36 windows | 0.925 / 0.912 (D-B-049); 0.931 (D-B-080) |
| EXP143v2 CoAtNet-2 rmlp (10-09) | 96 slices at native 384 px, 320 px centre crop, 94 windows, mirror TTA, all 4,347 studies, L2 labels (0.75 Dread + 0.25 aquila) | **0.946 / 0.934** (D-B-087) |
| EXP155 (10-10) | EXP143v2 + OAI supervision on 3 targets | 0.946 / 0.933 (D-B-101) |
| ConvNeXt-T/S partners (10-10..13) | 224 px, mixed 140 + 100 mm centre crops, 12 per-finding queries | local only: mixed scale +0.003 (D-B-117), finding queries +0.0017 (D-B-126); never scored alone |
| K1 ConvNeXt-S (10-17..22) | heliosli's public compact trainer and renderer (described below); trained on fold 0 only (3,190 studies, ≈73 %) | K2 v5 alone 0.944 public (D-B-219; handoff.md:46) |

K1's renderer (D-B-243): views "v4", 3-slice 3.5 mm slabs, 384 px render with a 352 px crop, joint/patella anchors from heliosli's anatomy.pt (D-B-265: built on SKM-TEA), a side model, attention pooling, at most 64 windows per study in training. K1 was otherwise used only inside P blends.

**Score discrepancy.** Your submission list shows no score for the K1/K2 standalone entries. Our records say Taeyang reported K2 v5 (ref 57381398) at 0.944. Please check which entry that is.

## 2. Why no own model got near 0.95, and the kill tests

The kill tests:
- H-B-151, soft-label CoAtNet: 0.944 vs its parent's 0.946 → DEAD (D-B-187).
- H-B-157, R-KD (public R distilled): 0.948 vs R's 0.949 → DEAD (D-B-191).
- H-B-156, ConvNeXt-S in the EXP143v2 pipeline: 0.926 → DEAD (D-B-190).
- H-B-177, ConvNeXt-S with its own recipe: 0.934 → DEAD (D-B-202).

The private scores bear these out: 0.927, 0.932, 0.911 and 0.924, all at or below EXP143v2's 0.934. **The verdicts were right in the narrow sense.** But every test changed the labels, a teacher or the backbone, and kept the same unanchored slot inputs. None of them tested geometry.

The D-B-202 cause analysis (10-16) did name the gap correctly: "not the architecture… but the data contract." That matches the top teams' write-ups.

K1 then tested that idea, but only partly:
- It was trained on fold 0 only, for 10 epochs.
- On report rulers it looked equal to the R7 fold-0 model (D-B-218: Dread +0.013, the other three within 0.003), yet it scored 0.944 public.
- Its own registered rule was "dead ≤ 0.946" (plan-snapshot-20261018-1609.md:322). No record shows that rule being applied. K1 simply carried on as a blend member.
- An all-data K1 was trained (D-B-255) but was only ever read inside blend C, at 0.957 (D-B-263). It was never scored alone.

## 3. Building on P, and the 0.15–0.25 own share

How the own share fell:
- 10-07: anchor 0.8 + own 0.2 gave 0.942, against 0.943 for the anchor alone (D-B-031).
- 10-09/10: own weight was large while our model was near the public one: blend3 (⅔ own) 0.951 (D-B-105). D-B-080 said the main lever was "a near-equal own partner".
- 10-12, once P956 (heliosli's 0.956 notebook) arrived: astra-3 registered C85 = 0.85 P + 0.15 own, and the weight grid it searched for own was only {0, 0.15} (archive-to-20261012:5060, :5096).
- The 10-15 type consults then put "own-weight tuning" on the STOP list (archive-20261013-14:2519).

Raising the own share was tested several times:
- 0.30: 0.955, a tie (D-B-167); with R7, 0.956 vs 0.957 (D-B-179). Both changed several things at once.
- 0.25: 0.957 vs 0.958 (D-B-246).
- A per-target vector up to 0.5: 0.957 (D-B-251).

The conclusion was that the leaderboard penalises own weight above about 0.15–0.20, so we kept a flat 0.20 and tried to improve K1 instead. One display step is inside the ±0.007–0.010 public noise (D-B-016), so these tests could not really decide the question. Even so, the conclusion was **correct given how strong our model was**: the 63rd-place team only gained from own weight 0.5 once their own model matched the public stack. The real miss is upstream.

## 4. Anchored crops, segmentation, whole-study reading, resolution

- **Segmentation (10-08):** a U-Net route was deferred: no clean knee checkpoint, a sequence gap, little runtime margin (D-B-045).
- **S6+ (10-12):** our own ConvNeXt-S with anatomy crops was built and passed 71 tests; anatomy.pt was downloaded. Estimated cost was a 254 GB cache, with H: at 51 GB free, and 31–92 GPU-h per fit. Taeyang asked to hold it and work on the highest-gain items first. The gain-priority consult then said "S6+ stop (blend Δ +0.0003–0.001)". It was dropped at 14:50 (archive-to-20261012:5192, :5216, :5574).
- **Our crop experiments:** these were geometric offsets of about ±18 mm from the knee centre, not segmentation-placed, and they *replaced* the full field (D-B-192, −0.004). The variant that added views to the full field (D-B-198: +0.0012; Lateral OA +0.011) and the Lateral OA expert (D-B-271: +0.0022) both failed their gates.
- **Laterality:** canonical right-knee rendering was dropped on report rulers, even though the OAI image-read panel showed +0.0081 (D-B-164).
- **Resolution:** native 320 px vs 224 px gained on all four rulers on two folds (D-B-178/184: macro +0.002–0.003). It was blocked by the all-cell floor rule.
- **Whole-study reading:** at test time this was done (94 windows; K1 stride 2 gave +0.001, D-B-247). In training, 38 % of bags missed a view (D-B-121). A full-bag objective lost as a replacement (D-B-264) and gained +0.0014 as a fourth member (D-B-278).

## 5. Compute

What we had:
- local RTX 4070 Ti SUPER, limited by reads from the G: HDD;
- Vast 3090 (10-09..12);
- Vast 4090s (10-15, and 10-20..22 until the credit ran out);
- Kaggle T4×2, 30 h a week; hidden reruns are free (D-B-059);
- Colab T4 lane admitted (D-B-090);
- RunPod prepared but stopped in consume-queue mode.

Cloud spending was planned against a KRW 100k ceiling (D-C-023).

The costs were small: EXP143v2 all-data 4.0 h (D-B-087), ConvNeXt-T 25 min per fold, K1 2.4 h per fold on the 4090 against about 13 h locally (D-B-255). GPUs also sat idle: the 4070 and the 3090 on 10-12 (archive :5581), and both 4090s stopped "no queued work" on 10-15 (D-B-190/192). **Compute was a friction (storage and I/O), not the bottleneck.**

## 6. How 10-13..10-22 was spent

About 125 discoveries, D-B-158 to D-B-282, by my own rough classification:

| Area | Count | Share |
|---|---|---|
| Blend composition, P-branch members, leaderboard readings | ≈52 | 42 % |
| Serving, runtime, parity, guards | ≈26 | 21 % |
| Own-model training and inputs | ≈35 (18 on the K1 line, mostly as a blend member) | 28 % |
| Labels | ≈4 (D-B-166, -180, -235, -259) | 3 % |
| Scouting and decision records | ≈8 | — |

Submissions: about 49 refs, of which 6 were own-model-only (4 kill tests plus 2 K2) and about 43 were P-based blends.

## Misses

**M1 — anchored geometry adopted late, and only as a blend member.**
- Top teams: anchored local views at 336–448 px carried their submissions.
- Us: S6+ was stopped on 10-12 (above). K1 came only on 10-16/17, trained on fold 0 and given 0.20–0.25 weight.
- Cause: judgement plus a process rule. Own models were scored by their diluted value in the blend (×0.15 → +0.0003). Consume-queue mode (10-10) allowed "≥0.954 public-weight routes only", and plans were frozen against a 10-18 member freeze (D-B-133).
- Confidence: high.
- Falsifier: an all-data or 5-fold K1 scoring ≤ 0.946 alone.

**M2 — the rulers could not see geometry or laterality gains and over-credited K1.**
- Top teams: judged on image-read labels, within-site AUC, or external panels.
- Us: report rulers plus all-cell floors rejected D-B-164, -184 and -198, and said K1 ≈ R7 on report rulers when the leaderboard did not agree (D-B-218 vs 0.944; D-B-259).
- Cause: wrong ruler.
- Confidence: medium.
- Falsifier: the rejected arms also losing when scored on image-read labels.

**M3 — no standalone confirmation, and a kill rule we did not apply.**
- Top teams: a single 5-fold own model.
- Us: the all-data K1 was never scored alone, and K2's 0.944 against "dead ≤ 0.946" was not acted on.
- Cause: process rules ("Kaggle only for submissions", 10-13 16:2x) and treating submission slots as scarce.
- Confidence: medium.
- Falsifier: records showing an explicit decision to keep K1 despite K2.

**M4 — the late effort went into serving work.**
- Us: §6, D-B-242–281.
- Cause: judgement. 0.958 public ties made it look like there was still progress to make.
- Confidence: high.
- Falsifier: none really; the counts are descriptive.

**M5 — the strategic claim was withdrawn early.**
- On 10-07, D-B-013's line "the medal path runs through our own models… largest fixable lever" was withdrawn under cross-check as "a strategy judgement".
- Cause: the review process was built for narrow claims and filtered out direction calls.
- Confidence: medium-low.
- Falsifier: a later plan that restored own-model primacy and was then deprioritised for reasons based on evidence.
