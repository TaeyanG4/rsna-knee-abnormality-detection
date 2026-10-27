# Post-mortem — RSNA Knee Abnormality Detection

**Version 1.1.0** · B (Claude Code) for Taeyang, 2026-10-25 KST (1.1.0: 2026-10-28, final verified board and the
second solution collection, [top-solutions-20261028.md](top-solutions-20261028.md)) · Inputs:
[top-solutions-20261024.md](top-solutions-20261024.md), our Kaggle submission list (134 entries, private scores),
the preliminary private leaderboard, and four investigator reports that read discoveries.md, the handoff
archives, plans and memory ([evidence/](postmortem-20261025-evidence/)). An independent fact-check of the
draft (about 58 claims OK, 21 corrected or softened) was applied in this version.

Team ranks such as "72nd" or "38th" below are the *preliminary* ranks used as labels in the first collection;
the final ranks are mapped in [top-solutions-20261028.md §1](top-solutions-20261028.md#1-final-verified-standings).

## 0. Result and what the numbers already say

- **Private 0.943, rank 297 / 6,300, silver** on the final verified board (first team of the 287-team 0.943
  band; silver line rank 315). Public 0.958 (rank 625). Preliminary (10-23): 312 / 6,425, silver line 321; Kaggle
  then removed 125 teams.
- **The best private score among all our scored submissions (119 of 134) is 0.943**, first reached on 10-16.
  From 10-16 to 10-22 we made 34 submissions; 29 scored, 21 of them at exactly 0.943 and the rest at
  0.940–0.942 (plus two own-only tests). The final pair choice cost nothing; the last seven days of variants
  never moved the private score above 0.943.
- **Own models alone peaked at 0.948 public / 0.935 private** (B172, an own-only blend, 10-10); the best single
  own model was EXP143 v2 CoAtNet at 0.946 / 0.934 (10-09). K1 standalone (K2, 10-19) scored 0.944 / 0.931.
  Top teams' single models reached 0.957–0.958 private; the 72nd's single 5-fold model reached 0.949.
- Private progression: 0.923 (09-11) → 0.936 (10-09) → 0.940 (10-11) → 0.942 (10-13) → 0.943 (10-16) → flat.
- First submission 09-11, five to six weeks after the start (Kaggle enabled the page on 08-05; the overview
  timeline gives Jul 30); no records 09-17 → 09-30.

Score → rank on the final verified board: 0.944 = 204–296 · 0.946 = 124–160 · 0.948 = 74–98 · 0.949 = 59–73 ·
0.950 = 49–58 · 0.952 = 31–40 · 0.954 = 22–25 (gold line 22) · 0.956 = 11–15.

## 1. What we missed, one by one

Each item: what the better teams did · what we did (records) · why we missed it · estimated private value.

### M1. Image-read labels never became the main label (largest miss)

- **Top teams:** an image-reading LLM as the label anchor (gold-58 about 0.91–0.95 vs 0.89–0.91 report-only),
  fused per finding, LB-checked; worth +0.006 to +0.016 private (72nd, Northstar, 5th).
- **Us:**
  - D-B-074 (10-09) showed the lever first: with images + report label + rules, the Opus reader beat the report
    label on gold-58 by +0.0148 (fresh-38 +0.0189); the Sol reader's +0.0081 was not significant. On 10-13
    astra-4 challenged the entry: with no rules-plus-label control, the image contribution is not isolated.
  - aquila_v1 was built the same day for 4,405 studies: Astra variant F, 15 central slices, report text and the
    flight_v4 label, our relabel v1.1 rules (D-B-078). Gold-58 macro 0.909 vs flight 0.899.
  - It only ever entered as a 25 % share (L2, D-B-083/095/123). The highest image share that reached the LB was
    that 25 % (EXP143 v2).
  - The full Opus/Sol reads (cygnus_v1, lyra_v1) were never produced; Opus over all studies (~$2–4k) was judged
    not justified (D-B-078), Sol E was never run.
- **Why:**
  1. *Mostly the wrong ruler.* L1 (aquila only) and the 60 % per-target share (D-B-166) were judged mainly on
     report-derived rulers, which cannot see an image-label gain. Gold-58 was also slightly negative (L1 −0.0098;
     D-B-166 −0.005 on seed 42, −0.0013 pooled), inside its ±0.012–0.016 noise. L1's MCL −0.054 suggests a
     *full* replacement was partly wrong, which is why the top teams fused per finding. For D-B-166 the registered
     rule actually read GO_RULE_MET (astra-1 correction, discoveries.md:3824); B dropped it as a holistic call and
     it was never reopened.
  2. *No LB check.* B proposed a paired public check of aquila (EXP150) on 10-09 08:43. Taeyang declined it to
     save submissions, noting that local validation was possible (handoff-archive-to-20261012.md:893). B had
     recorded on 10-07 (D-B-042) that label changes cannot be judged by own-label references, and should have
     said so at that moment; it did not.
  3. *Weaker reader set-up than the top teams.* The full run kept our relabel v1.1 rules (D-C-002 had already
     listed wording gaps against the official definitions), although the 10-09 06:35 Astra F gold-58 diagnosis
     showed misses on MCL (7/9 positives) and PF OA (8/21); that event chose "keep the validated prompt"
     (handoff-archive-to-20261012.md:772). Also 15 central slices instead of the study, report-aware only (the
     "give the reader everything" rule), no report-blind arm.
  4. *Targets in the one per-finding test.* D-B-166 raised the image share for effusion and synovitis but also for
     all three OA compartments, where the 8th gave the image read only 1/6–1/2; fracture, Baker's and contusion,
     where image reads gained most for the top teams, were not included.
  5. *Label work stopped on 10-12/13* (pre-freeze schedule, Codex token shortage, consume-queue mode), and on
     10-19 D-B-235 classed re-reading as overlap with settled work.
  6. Re-levelling report states to gold-58 rates (38th, 5th) was dropped on 10-09 14:05 (H-B-36) as a "weak
     detour beside existing image-checked labels and EXP149" — astra's plan-review verdict, adopted by B.
- **Value:** about +0.006 to +0.010 private on our own line.

### M2. Our rulers could not see the changes that mattered (root cause of M1, M3 and M4)

- **Top teams:** judged label and site changes on image-label OOF, within-site AUC, external panels, sealed
  folds and per-finding LB probes.
- **Us:**
  - Almost every gate used report-derived rulers: JEV, Dread, consensus, flight, the 976-study holdout. Lenses
    L1–L6 were report-derived, and L7, the only image lens, was usually ineligible under the ancestry rules.
  - We wrote the limitation down early. D-B-042 (10-07) says "label changes must not be judged by any own-label
    reference"; D-B-112 says no report ruler is adequate for synovitis or contusion. The one image-read ruler we
    built (the OAI MOAKS panel) was usually ineligible under the ancestry rules, and no locked image panel on
    competition studies was built.
  - The evidence ledger never found an LB-tracking ruler and stopped after 10-14. D-B-259 (10-21): the report
    lenses got one of the two known LB-ordered pairs wrong (the other pair right; the wrong one is confounded
    with stride).
  - Per-finding LB probes were proposed (fable P1 → H-B-120, waiting for Taeyang's go) and closed on 10-13 18:04
    in astra-3's re-prioritisation, applied by B, because the new rule of that day said "no
    probes/diagnostics/pre-qualification" kernels. The probe needed kernel variants and visible runs, so our own
    quota rule did exclude it — and with it the one ruler that could judge label changes.
  - Folds were component/hash-grouped, never site-grouped. The scanner holdout required by ADR 0004 was never run.
- **Why:** process rules (ancestry eligibility, no gold tuning, our own probe/diagnostics ban) plus no investment
  in an image-read validation panel on competition studies.
- **Value:** not a score by itself; it decided M1, M3 and M4.

### M3. Site / hospital reporting habits were never removed in training

- **Top teams:** a bias term only in the training loss, dropped at test: +0.003 to +0.007 private (38th one-factor
  +0.003 private / +0.006 public overall, 63rd +0.005, 72nd +0.007), and 33rd +0.014 private from its no-trap
  baseline. 38th found that shifting outputs at test time recovered only about half of the gain.
- **Us:**
  - One post-hoc test, D-B-002 (10-06): cross-fitted per-group score offsets (field, scanner group, manufacturer,
    cached mask) on stored predictions, scored against Dread labels. Dread carries the habit itself, so the test
    was biased against removing it. All variants but one (cached mask, label-aware, +0.0013/+0.0015) lost; the
    first verdict "closed as a cheap lever" was narrowed on 10-07 to "these two estimators did not help; other
    forms untested", and no other form was tried.
  - The 10-02 forum triage had seen that a metadata-only model scored 0.652 on random folds vs 0.598 on
    scanner-grouped folds ("site memorisation"), yet D-B-013 (10-07) recast it as "no shortcut".
  - No per-site or per-language positive-rate analysis exists in our records; a training-only term was never
    proposed. (D-B-252 screened a PatientSex prior post hoc — negative, not adopted.)
- **Why:** not considered after D-B-002, plus the wrong ruler.
- **Value:** about +0.003 to +0.007 private.

### M4. Our own model stayed at 0.934 private

- **Top teams:** a strong own model carried the submission: whole-study reading, mm crops at 336–448 px,
  anatomy-anchored local views, per-finding readout that keeps the side.
- **Us:**
  - EXP143 v2 (10-09) was a copy of the public Raptor recipe: centre crop, no anatomy anchoring, no laterality
    handling. No later own model beat it standalone.
  - The four kill tests of 10-14..16 changed labels, teachers or backbones, never geometry. Their private scores
    (0.911–0.932) confirm the verdicts; they simply did not test the right thing.
  - S6+ with anatomy crops was built and passed 71 tests, then stopped on 10-12. Its 254 GB cache was planned for
    C: (628 GB free), so disk was not the block. Taeyang asked to hold design E and work on the highest-gain items
    first, and the gain-priority consult valued it by its diluted blend share (blend Δ +0.0003–0.001 = member Δ ×
    ≈0.4–0.55) (handoff-archive-to-20261012.md:5192, :5216).
  - Laterality-canonical rendering was dropped on report rulers although the OAI image panel showed +0.0081
    (D-B-164). Native 320 px over 224 px was blocked by an all-cell floor rule (D-B-178/184).
  - K1 v1 collapsed on 10-17 (D-B-214); the 10-epoch v2 actually used was trained on 10-18 (public anchored
    renderer, fold 0 only). K2 (10-18, scored 10-19) alone reached 0.944 / 0.931, below K1's own kill rule
    ("dead ≤ 0.946"), which was not applied. The all-data K1 was never scored alone.
- **Why:** judgement (own models valued by their diluted blend share), process (consume-queue mode blocked new
  plan items; the 10-18 member freeze), wrong ruler. Compute was friction (I/O, cost caution), not the
  bottleneck: EXP143 v2 all-data took 4 h, and GPUs sat idle on 10-12 and 10-15.
- **Value:** about +0.002 to +0.005 private on top of M1 and M3.

### M5. The last week went into +0.001 work on a saturated family

- **Top teams:** public-recipe blends plateaued at private 0.942–0.944 (220th, 246th, 412th, 478th, 507th,
  Northstar); the public-model authors themselves finished 0.942–0.944.
- **Us:**
  - From 10-13 to 10-22 (about 125 discoveries): about 42 % blend composition, 21 % serving/runtime/guards, 28 %
    own-model work (mostly K1 as a blend member), 3 % labels. About 43 of 49 submissions were P-based blends.
  - The final pair X and F are both P-dominant (F is X's flat-P twin), against the biohub lesson of one
    candidate from a different family. It did not cost us here, because every family member scored 0.943.
- **Why — the right diagnosis was on record and not acted on:**
  - 10-13: fable wrote that top teams "own stronger single models trained on better labels", that "our label
    work stopped at L2", and that our verification was "excellent at not fooling us and poor at moving the
    score" (.local/strategy-1013/fable/REPORT.md:9–17).
  - 10-16 01:43: fable proposed L3 labels plus an own-heavy pivot. At 01:59/02:12 fable itself re-estimated it
    (3–5 %) after Taeyang recalled that aquila already existed and its dose-response was ≤ 0 on report rulers —
    the M2 trap again.
  - 10-16 01:57: in one event the biohub lessons were re-shared (their gist: a large gap is structural, not a cue
    for +0.001 steps), Taeyang set a "+0.001 every 1–2 days" target, and B proposed daily stacking of small
    changes. Fifteen minutes later the debate settled on "daily single-change tests".
  - The 0.958 public ties made it look as if there was still room to gain.
- **Value:** it took the time that M1/M3/M4 needed.

### M6. Process overhead crowded out research

- **What happened:**
  - About 55 % of tagged discoveries are infra/release/runtime/parity, and about 84 % of handoff events are
    process or infra.
  - We built 963 tool files, 551 tests, about 20 gate versions, dispatch v3.2, 17+ Codex lanes and an 8-step gate.
  - Consume-queue mode (10-10 19:57) froze new plan items about 15 h after L2 was adopted (10-10 05:12), with the
    full-read plan (Sol B + Astra F) still fresh.
  - Strict byte-diff gates on hidden-proven parents and "plan = candidate" favoured blend patches over a
    multi-day label → retrain → serve chain.
  - Taeyang, 10-11 16:20: five hours of only verification fixes.
- **Why:** each rule answered a real failure (deadlocks, hidden-run errors, token exhaustion), but together they
  raised the cost of exactly the large changes the gap needed.
- **Value:** indirect; the main reason M1–M4 were not done in the 13 days after 10-09.

### M7. Late start

- **What happened:** first submission 09-11 (five to six weeks in); no records 09-17 → 09-30; 10-01–02 went into
  the KneeOps web app. The effective research window was about three weeks.
- **Why:** not recorded (low confidence).
- **Value:** it set the ceiling. The biohub lesson asks for big changes 4–6 weeks before the end, and we
  discovered the main lever 13 days before the end.

### M8. Forum signals were read for downloadable assets, not methods

- **What happened:**
  - The LLM-label benchmark thread (747658) was read on 10-16 and closed as "no new production label asset
    authenticated". A "single sub 0.960" thread was closed as "no usable authenticated member or recipe".
  - No site-habit thread appears in the scout records.
  - The thread that did change our final recipe was a public-weights one (749936, 0.958).
- **Why:** scout design (an "is there an asset to plug in?" lens) plus token limits.
- **Value:** indirect; the method signal that single models at 0.960 and image-aware labels existed was
  available in October.

## 2. Root causes

| Cause | Items | Note |
|---|---|---|
| **Wrong ruler** — report-derived validation judging label, site and geometry changes | M1, M2, M3, M4, M5 | The single most important cause; we recorded the limitation on 10-07 and kept gating on it |
| **Process rules that blocked large changes** — consume-queue, strict byte-diff gates, plan = candidate, no gold re-levelling, report-aware reader rule, our own probe/diagnostics kernel ban, duplicate-work rule | M1, M2, M4, M6 | Each rule was reasonable alone |
| **Judgement** (B, reviewers and Taeyang) — holistic DROP over a GO rule, saving submissions instead of an LB check of aquila, own models valued by diluted blend share, direction calls withdrawn, "+0.001 per 1–2 days" target, public ties read as progress | M1, M4, M5 | The correct diagnosis existed on 10-13 and 10-16; B knew the ruler limit (D-B-042) and did not press it |
| **Not considered** — training-only site bias, report-blind readers, per-finding probes as a ruler | M1, M3 | |
| **Resources** — Codex tokens, disk space, cost caution on full reads | M1, M4, M6 | Friction, not the binding constraint |
| **Time** — late start | M7 | Set the ceiling |

What went right: D-B-074 found the main lever early; D-B-042 correctly described the rulers' limit; fable's
10-13 report described the gap correctly; hidden runs were robust in the final week; selection lost nothing.
The system generated the right insights — the failure was in acting on them.

## 3. Where we could have finished without these mistakes

Gains do not add exactly and come from other teams' pipelines; ranges are judgement, not measurement.

| Scenario | Changes | Private estimate | Rank on this board |
|---|---|---|---|
| A. Minimal fix, same timeline | Submit L1 / per-finding image-heavy labels on EXP143 v2 and blend at real weight | 0.945–0.947 | ≈ 99–203 |
| **B. Realistic, same timeline (act from 10-09)** | A + training-only site bias + OOF rounds re-anchored to the fused label + anchored crops on our own model; blend with P at ~0.5 | **0.948–0.951** | **≈ 41–98** (silver) |
| C. Strong execution, same timeline | B done well (full image reads with the best reader, per-finding fusion, 5-fold K1-class model, different-family final pair) | 0.952–0.954 | ≈ 22–40 (gold line at 22) |
| D. Full window (start in August) | C with 6 more weeks | 0.954–0.956 | ≈ 11–25 (gold) |

Anchors for B: the 72nd's single-model ladder went 0.932 → +site bias 0.939 → site-bias teachers 0.942 →
attention KD 0.943 → +image labels 0.949 — our EXP143 v2 (0.934) sits at their starting point. The 63rd reached 0.948–0.949 when its
own model was strong enough to take a 0.5 share. 49th: own 0.949, blend 0.951. Anchors for C/D: 32nd (0.952),
22nd solo (0.954, 56 networks on rented 4090s), 21st solo (0.955).

**Most likely outcome without the mistakes: roughly rank 40–100 (private ~0.948–0.951), with gold (top 22)
reachable only with strong execution or an earlier start.** Ranks are on the final verified board (10-28).

## 4. Lessons for the next competition

1. Build one ruler that can see the target before gating on anything: an image-read or external panel, sealed
   folds, within-site AUC, and per-finding LB probes. Never let a ruler built from the training labels judge a
   change to those labels.
2. When the gap to the top is large (here 0.015), work on supervision and the core model, not blend weights.
   Re-read closed directions when a better ruler exists.
3. Score own models standalone and keep a kill rule that is actually applied; value them by standalone
   strength, not by their diluted share in a blend.
4. Check for site / scanner / language habits in the labels on day one; add a training-only bias term.
5. Keep process rules proportional: allow one large multi-day change outside the patch gates, and do not let
   token or freeze rules stop the highest-upside line.
6. Read forum threads for methods and score levels, not only for downloadable assets.
7. Start the main line in the first two weeks.

Added after the second collection (10-28; evidence in [top-solutions-20261028.md §4](top-solutions-20261028.md#4-feedback-against-our-record)):

8. Before judging an image reader, give it the whole study or slice-picking tools; our 15-slice aquila read
   probably understated the lever (2nd: a partial-view pilot misled them too; whole exam +0.043 on gold-58).
9. Judge label sets by log loss and confident errors as well as AUC, and calibrate them.
10. Use the expert-labelled studies as training data too (normal weight, or a short fine-tune merged back),
    checked on a split of them, not on report-label OOF.
11. Validate an external dataset's label mapping per finding before judging the dataset (OAI paid for 13th and
    10th after re-mapping).
12. Give the weakest findings (here lateral OA, PF OA, synovitis) their own label and model track early.
13. Point the agents at the science (labels, ROI boxes, experiments), not mainly at process.

## 5. Change log

- 1.1.0 (2026-10-28): final verified board (297 / 6,300; silver line 315), score → rank map and scenario ranks
  recomputed on it; note that team labels are preliminary ranks; lessons 8–13 from the second collection.
- 1.0.0 (2026-10-25): first version, with the independent fact-check applied (decision ownership for the 10-09
  LB check, S6+ and the 10-16 target; H-B-120 closed under our own rule, not a misreading; 33rd baseline; D-C-002
  vs the 10-09 diagnosis; D-B-002 groups and narrowed verdict; D-B-166 targets; D-B-259 pair result; K1 dates;
  consume-queue scope and timing). Same day, before publishing: last-week count corrected (34 submissions,
  21 at 0.943, rest 0.940–0.942 — not "35 all at 0.943"); own-only peak is B172 0.948/0.935; start date wording.
