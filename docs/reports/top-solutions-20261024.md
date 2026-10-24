# Top solutions — RSNA Knee Abnormality Detection (collected 2026-10-24)

**Version 1.0.0** · Collected by B (Claude Code) for Taeyang on 2026-10-24 KST, one day after the close.
Per-write-up details: [top-solutions-20261024-details.md](top-solutions-20261024-details.md).

Purpose: step 1 of the post-mortem (feedback, review, what we did wrong). This page gathers what the
better-placed teams did; section 5 lists first observations against our own record, to be checked in the
review step. The short versions are README sections 12 (top-team strategies) and 13 (self-feedback).

## 0. Scope and method

- **Sources.** 25 solution write-ups linked from the leaderboard (Kaggle "Writeups"), plus 8 discussion posts
  (solution-like posts from ranks 30, 32, 38, 65, 507 and 1793, and the host's thank-you post).
  No write-up existed yet for private ranks 1–4, 6 or 9–20 at collection time.
- **How.** The leaderboard and write-up bodies were read through Kaggle's site API in the signed-in in-app
  browser (read-only; nothing was posted, voted or downloaded). Seven reader agents summarised the texts into
  one fixed template; numbers are as the authors wrote them.
- **Trust.** Write-ups are third-party claims, not verified results. Model names are copied as written. The
  private leaderboard was still marked *preliminary* by Kaggle at collection time.
- **Local raw copies.** The 25 write-up texts are kept for reference in
  `.local/archive/top-solutions-20261024/raw/` (local only — third-party text, not for publication).

## 1. Final standings snapshot (preliminary private leaderboard)

| Item | Value |
|---|---|
| Teams | 6,425 |
| **Our team (Taeyang: taeyangg4, lucas040)** | **private 0.943, rank 312 — silver**; public 0.958 (rank 653); 132 submissions |
| Gold line | rank 22, private 0.954 |
| Silver line | rank 321, private 0.943 |
| Bronze line | rank 642, private 0.942 |
| Density | 304 teams at private 0.943 (ranks 312–615; we are the first of that band); 1,322 teams at 0.942; 1,638 teams showed public 0.958 |
| Public → private, top 50 | mean drop 0.011 |
| Public #1 | Barry (songqizhou), 0.970 → private 7th, 0.957 |

Private top 10: 1 Team Kangaroo 0.960 (public 0.969) · 2 MURAD 0.960 · 3 DataTech Club 0.959 · 4 Ilya Kolb 0.958 ·
5 RB & Matheus & Tucker & Tony 0.958 · 6 CentralyDS 0.958 · 7 Barry 0.957 · 8 joinT fusion 0.957 ·
9 Parag & Ian & Dieter & CroDoc 0.957 · 10 MEJAL 0.957.

Authors of the public models that most blends (ours included) were built on finished in the same band as us:
heliosli 438th (0.943), goodpjw2008 281st (0.944), pilkwang 560th (0.943), nartaa 769th (0.942),
Dread Development 977th (0.942).

## 2. Source index

Private / public are the leaderboard values; "W" = write-up, "D" = discussion post.

| Priv. rank | Team | Private / public | Src | Main idea |
|---:|---|---|---|---|
| 5 | RB & Matheus & Tucker & Tony | 0.958 / 0.969 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/5th-place-solution) | Report labels recalibrated on gold-58 (fold-safe) → LLM + vision re-reads → OOF fusion; bone-segmentation-anchored crops as extra series; site-cluster folds |
| 7 | Barry (solo) | 0.957 / 0.970 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/7th-place-solution-with-no-external-training-data) | Three GPT-6 Astra image-aware reads (gold-58 up to .947); per-site synovitis shift; ½ OOF teacher; 2.5D label-query net with mm 3D positions; best single = public #1; no external data |
| 8 | joinT fusion | 0.957 / 0.968 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/8th-place-solution) | Image-first Astra labels for 4,406 studies, fused per finding with text labels; two isotonic OOF distillation rounds; up to 168 slices; BCE + ranking loss |
| 21 | Driz Lo (solo) | 0.955 / 0.964 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/21st-place-solo-gold-solution-llm-soft-labels) | Image-first LLM soft labels for all 4,349 studies via an image-server tool (gold 0.896 → 0.935); 0.75/0.25 self-distillation; public multiview 2.5D model |
| 22 | matsurih (solo) | 0.954 / 0.965 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/22nd-place-solo-gold-solution) | Astra montage reads (0.929) + Fable report reads (0.892); 50/50 OOF, refreshed 3×; femur-landmark crops; compartment classifier that cannot see the other side; 56 networks |
| 30 | knee-deep | 0.953 / 0.967 | [D](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751903) | Claude-drawn joint boxes → YOLO ROI crop per series (public 0.945 → 0.951) |
| 32 | Honda Gen-AI EXPs | 0.952 / 0.964 | [D](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751930) | 50 % report + 50 % VLM image labels; site-habit cells switched to the image read; evidence-slice attention loss; single model 0.943 → 0.960 public |
| 33 | hirokenn | 0.952 / 0.966 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/33rd-place-silver-solution-no-external-data) | Training-only site/acquisition "trap" branch (+0.014 private); report levels calibrated to gold rates; two-level MIL + 3D RoPE; no image-read labels |
| 38 | the fault in our stars | 0.952 / 0.964 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/the-hospital-is-a-variable-too), [D](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751924) | Report groups re-levelled to expert-positive rates ("not mentioned" ≈ 0.05: +0.012); 29-site train-only offset (+0.006); fixed-weight rank blend |
| 49 | Komil Parmar (solo) | 0.951 / 0.962 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/49th-place-silver-solo-whole-study-2-5d-reader) | "Describe, don't diagnose" report labels (0.899); text-anchored image-teacher rounds (→ 0.941); whole-study 2.5D slice-bag; per-finding column-inversion LB probe |
| 63 | Carl Pang | 0.949 / 0.960 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/63rd-place-solution) | Per-label teacher distillation incl. R6 OOF; site-cluster target debias (+0.005 private); DINO SSL on competition + fastMRI slices |
| 65 | kaggle & graduate trade-off | 0.949 / 0.960 | [D](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751920), [D](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751895) | Report-blind image readers only for weak classes, z-merged then remapped to the text distribution; IC-Zoom learned zoom |
| 72 | menphim (solo) | 0.949 / 0.960 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/72nd-place-training-only-site-bias-and-image-read) | Train-only site bias (+0.007 private); one teacher per public label table; montage + report labels (+0.006 private); final = one 5-fold model |
| 73 | Volkopat (solo) | 0.949 / 0.962 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/kneesplatnet) | Agentic image reader with crop/zoom/measure tools (self-hosted open model); KneeSplatNet |
| 74 | Amanie El-Hajj | 0.949 / 0.962 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/74th-place-knee-centred-views-llm-labels-and-oa) | "Not mentioned" ≠ "normal" (0.863 → 0.904); OAI-trained readers re-order labels; detector-placed local views; members chosen on external panels |
| 76 | rhoskeri (solo) | 0.948 / 0.963 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/rsna-knee-ai-challenge-solution) | Teacher–student generations after a head-only gold fine-tune; RadImageNet init (+0.008 public); per-scanner synovitis offset |
| 77 | Civitasmass (solo) | 0.948 / 0.960 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/multimodal-llm-relabeling-oof-self-distillation) | Joint report + montage relabel (+0.010 public); OOF distillation (gold 0.930 → 0.945); gold-58 never used directly |
| 102 | dalab | 0.947 / 0.963 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/102nd-place-local-llm-report-labeling-and-anatomy) | Local LLM report labels only; ROI U-Net + Gaussian anatomy priors; per-plane experts |
| 220 | Семён Давшиц | 0.944 / 0.960 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/220th-place-silver-gold-batch-fine-tuning-on-to) | Gold-58 batch sampling fine-tune on a public ensemble |
| 228 | Akhyar Ahmad (solo) | 0.944 / 0.961 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/228-solo-silver-subtract-the-report-style-prior) | Subtract a metadata-only model of the report labels (+0.002 to +0.003 private) |
| 243 | Chris Deotte | 0.944 / 0.959 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/243rd-silver-medal) | Frontier VLM labels from report + mosaic (0.9275 vs 0.888 report-only); quadrant pooling (+0.013 public, one model) |
| 246 | Eesh saxena | 0.944 / 0.960 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/246th-place-silver-report-labels-are-not-expert) | Public-weights blend + OA compartment rank decorrelation (+0.001 private) |
| 302 | BliverRigPåDoge (solo) | 0.944 / 0.959 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/solo-silver-302nd-3-min-video-of-my-solution) | Video only; own OAI-graded models added to the public blend |
| 412 | M. Ehtasham | 0.943 / 0.958 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/412th-place-bronze-per-finding-rank-blend-of-pu) | Public recipe + S6 arm at small weight; 4 of 5 late changes silently fell back on the hidden run |
| 478 | Prahas Duggireddy | 0.943 / 0.958 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/what-i-learned-from-training-my-own-models-on-radi) | Own report-label models at 15 % in the public blend; "labels learn whether the report mentions it" |
| 507 | yu_kuo14 | 0.943 / 0.957 | [D](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751949) | Whole deployed chain scored on gold-58 to pick a public-notebook variant |
| 671 | Northstar | 0.942 / 0.958 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/public-0-958-private-0-942-what-65-submissions) | 65-submission lessons; image-aware labels +0.016 private in a matched test; own models never helped the public stack |
| 977 | Dread Development | 0.942 / 0.958 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/the-original-raptor-12-findings-full-public-relea) | Raptor author: OAI made it worse; unfitted entry dropped as much as fitted ones |
| 1793 | Tumer Kaan Mutlu (solo) | 0.942 / 0.958 | [D](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751999) | Own 0.928 model at 10 % changed nothing; hidden-run fallbacks |
| 5049 | Don Whitteker | 0.818 / 0.823 | [W](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/mnist-to-mri-a-micro-mri-reader) | From-scratch NumPy micro reader |
| host | Po-Hao Chen (RSNA) | — | [D](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752012) | Thanks; winners "in the next few days"; nothing on private-set composition |

## 3. What the better teams did — common patterns

### 3.1 Labels that come from the images were the main lever

Report-derived labels measure whether the report *mentions* a finding, while the test labels come from
radiologists reading the images. Every write-up in the top 22 that used LLM labels had a model look at the
MRI, and most called the label step their largest gain.

| Team | Report-only labels (gold-58 macro AUC) | Image-aware labels | Effect on the model |
|---|---|---|---|
| 7th | .89–.91 | .935 (structured read) / .947 (verify-the-report prompt) / .952 after OOF teacher | best single 0.958 private |
| 21st | 0.896 (consensus of 4 public sets) | 0.935 (image-first, all studies) | single-model gold +0.02 to +0.03 |
| 22nd | 0.892 (Fable report read) | 0.929 (Astra montage); 0.935 combined | label sets 0.954 → 0.962 → 0.967 (10 findings) |
| 72nd | 0.903 (same prompt, report only) | 0.928 (montage + report) | +0.006 private in one step |
| 77th | ≈0.90 | 0.932 (joint report + montage) | +0.010 public on the same model |
| 243rd | 0.888 | 0.9275 (report + cropped mosaic) | — |
| 32nd | — | 50 % report + 50 % image read | single model 0.943 → 0.960 public |
| 8th | — | image-first read of 4,406 studies, image weight up to 5/6 | label work 0.937 → 0.954 public |
| Northstar (671st) | 0.893–0.906 | 0.911 | +0.016 private in a matched test |

How they did it:
- **One study per call, images first, host rubric in the prompt** (7th, 21st, 22nd, 72nd, 77th). Several made
  the image evidence win over the report (8th, 21st) or asked the model to check the report as a possibly wrong
  draft (7th).
- **Per-finding fusion, not one global share.** 8th weighted the image read 5/6 for fracture, synovitis and
  Baker's, 2/3 for effusion and contusion, 1/2 medial OA, 1/3 ACL and menisci, 1/6 PF OA, lateral OA and MCL.
  22nd used the image read for 10 findings, the report for lateral OA and a separate procedure for synovitis.
  77th: images helped effusion, contusion, Baker's and medial OA; reports were better for ligaments, menisci,
  PF OA and synovitis. 65th spent image readers only on the classes where text was weak.
- **Report-blind readers add uncorrelated errors** (65th: blind effusion 0.962 vs report-aware 0.926–0.931;
  report-aware readers copy the report). Image-only reads without the report were weak in other set-ups
  (243rd: 0.57–0.75; 77th: 0.877), so the gain is in combining sources.
- **Tools and resolution matter.** Crop/zoom raised one reader from 0.693 to 0.726 (74th); 73rd built an agentic
  reader with measure tools; 21st served native-resolution slices through a tool server.
- **Cost was manageable**: 72nd read all 4,407 studies in about 45 minutes with 128 parallel calls
  (≈21k tokens per study); 21st and 73rd report near-zero API cost (web client, self-hosted model).

### 3.2 Out-of-fold self-distillation, always re-anchored to the fixed label

Target = a·label + (1−a)·OOF prediction of an image ensemble, with a re-mix to the *original* fused label every
round (never to the previous round's target): 7th ½ (rank-mapped to keep prevalence), 8th 0.6/0.4 after
isotonic calibration, two rounds; 22nd ½, three label sets; 49th 0.3 text / 0.7 teacher, four rounds,
+0.013 of a +0.019 ensemble gain; 72nd 0.4/0.6 with one teacher per public label table; 76th ½ over five
generations after a gold head fine-tune; 21st 0.75/0.25. Two caveats recur: rounds pay only once the ensemble
beats its own labels (49th), and leaky report-OOF inflates the gain about 3× — use a sealed fold (49th).

### 3.3 Removing site / hospital reporting habits

Report-label positive rates vary strongly by site (PF OA 0.14–0.43 by report language, 33rd; synovitis 6 %–65 %
by scanner, 76th), and acquisition parameters predict the site with 98 % accuracy (33rd, 38th).
- **Training-only bias term, dropped at test**: 33rd +0.011 public / +0.014 private from its no-trap baseline;
  38th +0.006 public (adding the offset back at test lost score); 72nd +0.007 private, replicated across seed and
  resolution; 63rd site-cluster target debias +0.005 private.
- **Post-hoc variants were weaker but positive**: per-site synovitis shift (7th +.002 gold; 76th ≈+0.001 public);
  subtracting a metadata-only model of the report labels (228th +0.002 to +0.003 private).
- **Simple hygiene**: keep site-identifying header fields out of the inputs (22nd); group folds by
  scanner/site (5th, 49th); report-OOF with random folds is inflated 0.05–0.07 (49th) or 0.019 (5th).

### 3.4 Report labels re-levelled to the expert definitions

"Not mentioned" is not "negative" and "small" is not positive. 38th re-levelled each (finding, report group) to
the expert-positive rate measured on gold-58 with shrinkage (largest single step +0.012 public); 74th kept
"not mentioned" apart from "explicitly normal" (0.863 → 0.904); 5th found 199 low-grade MCL reports labelled
positive and 0 of 5 positive on gold, and calibrated fold-safely; 49th had the LLM *describe* and mapped the
description to probabilities in code (rubric verdicts collapsed to 0.809); 33rd mapped extracted levels to gold
positive rates.

### 3.5 Inputs: every slice, millimetres, anatomy-anchored local views

- Whole-study reading at test time: 49th (+0.009 gold; the only large single model gain besides input size);
  33rd and 74th every slice; 8th up to 168 slices (0.930 → 0.937 public).
- Crops in millimetres around the knee: 130–150 mm fields at 336–448 px (49th, 5th, 74th, 22nd).
- Anatomy-anchored local views placed by segmentation or detection: 5th bone segmentation (+0.003 private),
  22nd femur landmarks (TotalSegmentator femur class), 30th YOLO joint boxes (0.945 → 0.951 public), 74th RF-DETR
  meniscus/patella detectors, 102nd ROI U-Net with orientation, 32nd lateral zoom model on 3 columns.

### 3.6 Architecture: per-finding readout that keeps the side

Per-finding attention or queries over all windows and series are standard (7th label-query decoder, 49th 12
finding queries, 76th Query2Label). Additions that helped: 3D positions in mm (7th, 33rd 3D RoPE); a compartment
classifier with shared, mirrored weights that cannot see the opposite compartment (22nd; post-hoc subtraction
lost −0.007); quadrant pooling that keeps medial vs lateral (243rd, about +0.013 public on one model).
Backbones were not the differentiator: CoAtNet, ConvNeXt-T/S/B, EfficientNetV2 and NFNet all appear; bigger
backbones and foundation models rarely won (33rd: knee foundation model level with CoAtNet at 4× FLOPs;
49th: DINOv3, SigLIP, SSL all lost).

### 3.7 Validation and selection

- **Gold-58 cannot rank models** (bootstrap CI ≈ ±0.012–0.016; 49th, 74th, 77th, Northstar), and weights fitted
  on it lost on the board (21st, 49th, 74th, 412th), as did per-finding weights tuned on the public LB (63rd:
  −0.006 private).
- **Report-derived OOF is the wrong ruler for label changes and site effects** (5th: label sets tied at 0.873 CV
  but differed +0.008 private; 38th: report validation scored the "restore habit" test as a gain; 72nd: pooled
  report AUC fell while LB rose; 246th: a fix flipped sign on report OOF; 22nd rejected report labels as a
  reference).
- What they used instead: OOF against the image-read labels (22nd), within-site AUC (72nd), external panels
  never trained on — fastMRI+, MRNet, OAI hold-out (74th), sealed folds (49th), LB pairs with outcomes fixed in
  advance (38th, 7th), and per-finding LB probes: keep or invert one column to read that finding's public AUC
  (49th, 8th, 102nd).
- Pre-registered acceptance rules and measured noise floors were common (22nd, 49th, 77th, 38th).

### 3.8 Ensembling added little at the top

5th: the final ensemble tied its best single model; 7th: best single 0.958 beat the 4-model ensemble 0.957;
72nd: final = one 5-fold model; 8th and 22nd gained from *different views* (plane, compartment and crop
models), not from seeds or recipe tweaks (22nd: single recipe changes correlated 0.955–0.97 with the base and
moved nothing). Blends built on the public R6/Raptor/S6 recipe plateaued at private 0.942–0.944 (220th, 228th,
246th, 412th, 478th, 507th, Northstar, 1793rd); own arms helped only when they were 0.95-class and given real
weight (63rd: own weight 0.5 reached 0.948–0.949; small own weights stayed at 0.942–0.943).

### 3.9 Public → private

Drops were near-uniform: top teams 0.009–0.013, the public recipe 0.016–0.017 (63rd, 228th, Northstar, 977th).
The public leader fell to 7th. Per-finding probes put the largest private drops on synovitis, contusion,
lateral meniscus and PF OA (49th; 102nd estimated synovitis ≈0.906 → ≈0.828). Fixed-weight, mechanism-backed
entries held up; best-of-many public variants gave their gains back (228th, 76th).

### 3.10 Runtime and robustness

Shared DICOM decode and per-stage time budgets made large ensembles fit 2×T4 in 9 h (5th: 8 h 18 min; 22nd:
56 networks). Fail-soft legs in separate processes with renormalised weights (49th, 38th, 22nd); fail-visible
canaries exposed flaky hidden reruns (246th). Several late changes silently overran their guards and fell back
to the base, so they were never actually tested (412th: 4 of 5; Northstar: 3 of 5). Offline decoders for all
transfer syntaxes and a fix for studies that failed preprocessing were worth about +0.001 (77th).

### 3.11 What commonly did not work

OAI as extra *training supervision* (63rd six attempts, 977th, 1793rd; it helped only as pretraining, a 50/50
partner or an independent label ruler: 5th, 32nd, 74th); flip TTA on laterality-canonical inputs (49th −0.006,
77th); weights tuned on gold-58 or per finding on the LB; more seeds; bigger backbones; extra text-only LLM
readers (72nd: 0.86–0.89, no LB gain); rubric-verdict prompts (49th 0.809); synovitis inferred from effusion
(5th: confirmed 38 % of the time).

## 4. Where the gold zone came from (summary)

1. **Better targets**: image-aware LLM labels, re-levelled to the host definitions, fused per finding, then
   distilled with OOF teachers re-anchored each round.
2. **Removing site shortcuts in training** (training-only bias terms, site-grouped folds).
3. **A strong own model with the right geometry** (whole study, mm crops, anatomy-anchored views, per-finding
   readout that keeps the side) — strong enough that it, not a public blend, carried the submission.
4. **Rulers that can see the change**: image-label OOF, within-site AUC, external panels, sealed folds and
   per-finding LB probes, with pre-registered gates.

## 5. First observations against our record (to verify in the review)

These compare the patterns above with our own records. They are hypotheses for the post-mortem, not
conclusions; each needs checking against the cited discoveries. **Checked and superseded by
[postmortem-20261025.md](postmortem-20261025.md)** (fact-checked; corrects details such as decision ownership).

1. **Our final submissions sat in the public-recipe band.** X and F rank-blend the public branch P (Raptor
   R384/R224 + R6 flip) at 0.75–0.85 with our K1 models at 0.20–0.25
   ([project summary §3](project-summary-20261022.md)). Public 0.958 → private 0.943 (−0.015) is the drop of the
   public-recipe family, not the −0.009 to −0.013 of the top teams, and it matches 412th, 478th, 507th and
   Northstar. Raising our own share lost on the public LB (D-B-246/251), which fits the 63rd's finding that own
   weight pays only once the own model is about as strong as the public stack.
2. **We had the image-label signal on 2026-10-09 but never made it the anchor.** D-B-074: a vision LLM given the
   images + report label + rules beat the report label on gold-58 by +0.0148 (fresh-38 +0.0189); D-B-077: one
   exam per request mattered. The Astra image-checked set (aquila) then entered only as a 25 % share in L2
   (D-B-083/095/123). Pure aquila (L1) lost on the report-derived rulers (JEV −0.0058, MCL −0.054), and a 60 %
   per-target share for effusion, synovitis and the three OA compartments was dropped on report rulers and
   gold-58 while only the OAI image-read panel rose (D-B-166). The top teams kept the report for ligaments and
   OA compartments and gave the image read its largest weight on fracture, synovitis, Baker's, effusion and
   contusion (8th), and judged label changes on rulers that are not report-derived (§3.7).
   *Check*: exactly what aquila was (reader, inputs, coverage, blind or report-aware), whether any image-anchored
   label set was ever trained at full weight and scored on the LB, and whether our rulers could have detected
   the gain.
3. **Our pseudo-label test was anchored to report labels.** D-B-180 (R7 teacher, 50/50 with report labels) was
   NO_GO; the top teams' distillation re-anchored to image-aware labels each round, used a sealed or OOF
   teacher, and paid only once the ensemble beat its own labels (§3.2).
4. **We tested site offsets only after training.** D-B-002 found cross-fitted per-scanner score offsets lowered
   macro AUC. The variant that worked for 33rd, 38th, 63rd and 72nd is a bias term *inside the training loss*,
   dropped at test; 38th also found test-time shifting recovers only half. No training-only site-bias
   experiment appears in our discovery headings. *Check* the full text before concluding.
5. **Our rulers were report-derived plus gold-58.** JEV, Dread, consensus and aquila references plus the exposed
   gold-58 panel drove most gates. The top write-ups say these rulers cannot rank label changes or site effects
   (§3.7), which may explain items 2–4.
6. **Effort allocation (hypothesis).** Much of the last ten days went into serving our K1 members inside the
   public blend (shared render, strides, guards, runtime versions; D-B-242–281), where public 0.958 ties were the
   ceiling. The gold-zone gains came from label work and site handling, which were not on the final-week queue.

Suggested next step: a structured post-mortem that checks items 1–6 against the full discoveries, adds the
public → private results of our other final-day candidates, and fills the self-feedback section.

## 6. Change log

- 1.0.0 (2026-10-24): first collection — 25 write-ups, 8 discussion posts, leaderboard snapshot, first
  observations.
