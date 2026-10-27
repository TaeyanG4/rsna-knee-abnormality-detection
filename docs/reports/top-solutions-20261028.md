# Top solutions, second collection — RSNA Knee Abnormality Detection (2026-10-28)

**Version 1.0.0** · Collected by B (Claude Code) for Taeyang on 2026-10-28 KST, after Kaggle verified the final
leaderboard. Per-write-up details: [top-solutions-20261028-details.md](top-solutions-20261028-details.md).
First collection (25 write-ups, preliminary board): [top-solutions-20261024.md](top-solutions-20261024.md).
Post-mortem this feeds: [postmortem-20261025.md](postmortem-20261025.md) (v1.1.0 points here).

## 0. Scope and method

- **Final board.** Kaggle marked the final leaderboard as verified (`finalLeaderboardHasBeenVerified`) and removed
  125 teams (6,425 → 6,300), presumably for rule violations; no reasons were published. Every rank below is the
  final verified private rank unless marked *prelim.*
- **Sources.** The 16 write-ups that were linked from the final leaderboard but not in the first collection:
  private ranks 1, 2, 3, 6, 10, 13, 16, 19, 20, 31, 43, 127, 207, 383, 472, 478. Private ranks 4 (Ilya Kolb) and
  9 (Parag & Ian & Dieter & CroDoc) had still not linked a write-up. The 1st place post covers one member's part;
  the teammate's part (discussion 752474) and the second half of the 20th (Nikita Babych's post) were not read.
- **How.** Leaderboard and write-up bodies were read through Kaggle's site API in the signed-in in-app browser
  (read-only; nothing posted, voted or downloaded). Five reader agents paraphrased the texts into the fixed
  template of the first collection. Numbers are as the authors wrote them: third-party claims, not verified.
- **No raw copies** were kept this time; the details page and the write-up URLs are the record.

## 1. Final verified standings

| Item | Final (verified 2026-10-27) | Preliminary (2026-10-23) |
|---|---|---|
| Teams | 6,300 (7,490 competitors, 156,452 submissions) | 6,425 (7,683 / 161,275) |
| **Our team (Taeyang: taeyangg4, lucas040)** | **private 0.943, rank 297 — silver**; public 0.958, rank 625; 132 entries | rank 312; public rank 653 |
| Gold line | rank 22, private 0.954 | rank 22, 0.954 |
| Silver line | rank 315, private 0.943 | rank 321 |
| Bronze line | rank 630, private 0.942 | rank 642 |
| Private 0.943 band | 287 teams, ranks 297–583; we are the first of the band | 304 teams, 312–615 |
| Private 0.942 band | 1,297 teams, ranks 584–1,880 | 1,322 teams |
| Public 0.958 | 1,591 teams | 1,611 (1,638 at the deadline) |
| Public → private, top 50 | mean drop 0.011 | 0.011 |

Of the 125 removed teams, 15 had placed above us. The top of the board did not change: 1 Team Kangaroo 0.960 ·
2 MURAD 0.960 · 3 DataTech Club 0.959 · 4 Ilya Kolb 0.958 · 5 RB & Matheus & Tucker & Tony 0.958 · 6 CentralyDS
0.958 · 7 Barry 0.957 (public #1, 0.970) · 8 joinT fusion 0.957 · 9 Parag & Ian & Dieter & CroDoc 0.957 · 10 MEJAL
0.957.

**Score → final rank:** 0.960 = 1–2 · 0.959 = 3 · 0.958 = 4–6 · 0.957 = 7–10 · 0.956 = 11–15 · 0.955 = 16–21 ·
0.954 = 22–25 · 0.953 = 26–30 · 0.952 = 31–40 · 0.951 = 41–48 · 0.950 = 49–58 · 0.949 = 59–73 · 0.948 = 74–98 ·
0.947 = 99–123 · 0.946 = 124–160 · 0.945 = 161–203 · 0.944 = 204–296 · 0.943 = 297–583 · 0.942 = 584–1,880.

**First-collection sources on the final board** (prelim. → final): 5→5, 7→7, 8→8, 21→21, 22→22, 30→29, 32→31,
33→32, 38→37, 49→47, 63→61, 65→63, 72→70, 73→71, 74→72, 76→74, 77→75, 102→100, 220→212, 228→220, 243→235,
246→237, 302→287, 412→389, 478→453, 507→478, 671→639, 977→943, 1793→1737, 5049→4943. The first collection and
the post-mortem keep the preliminary numbers as their labels.

Public-model authors: goodpjw2008 268th (0.944), Heliosli 414th (0.943), Pilkwang Kim 528th (0.943), Dread
Development 943rd (0.942).

## 2. Source index (second collection)

| Rank | Team | Private / public | Main idea |
|---:|---|---|---|
| 1 | [Team Kangaroo](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/1st-place-solution) | 0.960 / 0.969 | One agent session per study with slice/crop tools, evidence before probability; a reading guide "trained" on gold feedback; agent-drawn anatomy ROI boxes → detector → per-finding box pooling; lesion boxes and report embeddings as training-only heads |
| 2 | [MURAD](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/2nd-place-gold-solution-vision-llm-soft-labels-an) | 0.960 / 0.969 | Native-language report reader (0.899) → whole-exam image read, ~277 images per call (0.942) → per-finding fixes, OOF mix, PF OA floor (0.956 on gold-58); six crop-diverse MIL members, hand-set weights never searched |
| 3 | [DataTech Club](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/3rd-place-solution-rsna-knee-abnormality-detectio) | 0.959 / 0.968 | Astra image + report soft labels, specialist re-reads for PF OA / lateral OA / synovitis; OAI expert + teacher targets 50:50; 36 full-data readers in fixed-budget groups with weak-three experts |
| 6 | [CentralyDS](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/6th-place-solution-knee-frame-tiles-pseudo-label) | 0.958 / 0.969 | Text labels + MOAKS-trained "second reader"; keypoint knee frame and anatomy-anchored 70 mm tiles; OAI + fastMRI noisy-student pretraining; same-scanner pair proxy AUC; 55 models with inference-only attention sharpening |
| 10 | [MEJAL](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/10th-place-solution) | 0.957 / 0.967 | Report LLM + VLM labels; OAI re-mapped to 7 soft targets with masks; five OOF distillation rounds with image-only Astra reads (+0.017 private in one round); noisy-OR rank post-processing |
| 13 | [Advanced CV](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/13th-place-solution) | 0.956 / 0.966 | Ordinal soft report grades (+0.015 private); agentic image + report labels (+0.005); OAI MOAKS rows mapped per finding (+0.008); TensorRT FP16 to fit 12 members |
| 16 | [Pa3฿aJluHа](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/16th-place-text-only-llm-labels-self-training-c) | 0.955 / 0.965 | Text-only: five open LLMs × 2 reads with the host rubric; ten self-training rounds; ConvNeXt-S grid tokens (+0.008 public); a retrained public S6 pipeline as the diverse partner |
| 19 | [AdeptusOverfiticus strike back](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/19th-place-solution) | 0.955 / 0.965 | Astra labels (with and without report, averaged); OOF ensemble distilled into one fast student that beat its teacher; MedSAM-anchored local crops; 512 px 1-slice CoAtNet |
| 20 | [YC & NB](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/20th-place-solution) | 0.955 / 0.967 | I3D-inflated 2D CNNs (0.930 → 0.947 LB) with per-label queries; gold-58 fine-tune merged by WiSE-FT (team +0.002 private); in-sample pseudo-label flywheel |
| 31 | [Honda Gen-AI EXPs](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/32nd-place-better-teachers-better-places-to-look) | 0.952 / 0.964 | 50/50 report + GPT-6 image teacher; site-habit cells switched to the image read; VLM-cited slices supervise attention; zoom models on three hard findings |
| 43 | [Effusion confusion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/45th-place-solution) | 0.951 / 0.963 | DINOv2 ViT with per-finding vectors routed by anatomical slot priors; label sets as rank re-orderings of one fixed distribution ("Refined" +0.005 private) |
| 127 | [Boris Konstantinovskii](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/129th-place-silver-llm-rubric-labels-slice-big) | 0.946 / 0.960 | Rubric-graded report labels with confidence weights; teacher → report anneal; compartment-routed pooling + slice BiGRU; 50/50 with the public stack |
| 207 | [Robson](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/215th-place-public-blend-radimagenet-field-strengt) | 0.944 / 0.960 | 0.75 public notebook + 0.25 own RadImageNet GRU model; transductive field-strength rank remap; warns that un-nested OOF teachers leak |
| 383 | [github/shepsci/kaggle-skill](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/three-experts-seven-changes-one-bronze-medal) | 0.943 / 0.958 | Public weights only: S6fall as a third expert, shift/mirror views, standardised log-odds instead of ranks; pre-registered keep/reject rules |
| 472 | [Jay Gautam](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/499th-place-bronze-0-943-private-4-family-rank) | 0.943 / 0.958 | Public four-family rank blend; checkpoint-provenance audit (Raptor's gold-58 AUC is selection-biased); weights picked by 600 split-half gold runs |
| 478 | [yu_kuo14](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/478th-bronze-a-variant-of-the-public-0-958-note) | 0.943 / 0.958 | Public 0.958 notebook with re-set Raptor weights and a 0.1 own reader, picked by a deployed-chain gold-58 check |

## 3. What the second collection adds

### 3.1 Image-read labels: confirmed at the very top, and the reader's view decides the gain

- Ranks 1, 2, 3, 10, 13, 19 and 31 all had a frontier model read the MRI. The new numbers sharpen *how*:
  - **Show the whole exam.** 2nd: one call with every slice (~277 images) took gold-58 from 0.899 to 0.942
    (+0.043); an earlier pilot that showed 44 of 134 slices had wrongly suggested images would not help.
    13th: images in the loop +0.034; a fixed ~95-slice set 0.928; letting the agent request slices only +0.003.
    1st: a median of ~15 images per study, but chosen by an agent with crop/zoom and a cross-series locator.
  - **Protocol matters as much as the model.** 1st compared reading protocols on gold-58: with the same model,
    switching protocol cut confident errors from 92 to 61 (GPT-6 Luna) and from 63 to 44 (GPT-6.1 Sol); the best
    model and protocol reached 0.936 with 35. A reading guide written by the agent from gold feedback
    cut confident errors to 19 out of fold at equal AUC. Labels were then judged by log loss and confident
    errors, not AUC alone, and temperature-calibrated on gold (k ≈ 0.77).
  - **Do not show the expert answers.** 2nd: the reader copied them, and a rule it wrote from them scored 0.80
    blind.
- **Text-only labels can still reach the gold zone** with a strong rubric and a strong model: 16th (0.955; five
  LLMs × 2 reads, ten self-training rounds) and 6th (0.958; text labels plus a MOAKS-trained second reader). Both
  authors still call labels their weakest area.
- **Per-finding source choice** recurs: PF OA half from the report reader (2nd: reader 0.960 vs Astra 0.930); a
  blind lateral pass for lateral OA (2nd: 0.856 → 0.898); specialist re-reads or models for the weak three —
  lateral OA, PF OA, synovitis (2nd, 3rd, 31st, 20th). A "not clear" PF OA call was positive in all 12 gold
  cases (2nd: floor 0.8).

### 3.2 OAI works when the mapping is validated per finding (revises the first collection)

The first collection listed OAI as training supervision under "did not work". The new write-ups show where it
worked: 13th mapped MOAKS per finding with masks and reliability-weighted mixing (+0.008 private, +0.009 LB in a
second track); 10th re-mapped OAI to 7 soft targets after the threshold mapping failed (+0.002–0.003); 3rd mixed
expert and teacher targets 50:50 (+0.001 both boards, and more OAI was not better); 6th used a MOAKS-only model
zero-shot as a label "second reader" (effusion 0.968 AUC on gold) and OAI + fastMRI noisy-student pretraining
(about +0.003 public); 20th used OAI tissue masks as an auxiliary segmentation loss for OA crops.

### 3.3 Anatomy decides the readout

1st: agent-drawn per-finding anatomy boxes, a detector at test time (IoU 0.71), box pooling with in-box attention
(gold 0.932 → 0.943 over box pooling, the 112 mm crop, in-box attention, calibrated labels and two training-only
heads); study-level, 2.5D and 3D variants added nothing once boxes were
used. 6th: a keypoint knee frame (side from anatomy, never from DICOM tags) and anchor-sampled 70 mm tiles.
19th: frozen MedSAM bone masks to anchor local crops. 127th: compartment-routed pooling (medial findings see only
the medial half). 20th: I3D-inflated 2D CNNs with one query per label (LB 0.930 → 0.947 for the same backbone).
16th: grid tokens instead of global pooling (+0.008 public).

### 3.4 Teachers and gold-58 as training data

- 20th: gold-58 fine-tune of every leg, merged with WiSE-FT (α 0.7), was their largest single step (team
  0.959 / 0.947 → 0.962 / 0.949); in-sample full-train teachers beat OOF teachers.
- 2nd, 3rd, 10th, 31st put the 58 expert studies into training with their expert labels; 3rd refit its members
  with gold after freezing the ensemble.
- 19th: an OOF ensemble distilled into one student beat its teacher (0.951 vs 0.948 private) in 1/15 of the
  runtime, which freed time for heavy diverse members.
- Leakage warnings: OOF teachers must be nested when gold studies sit in training folds (207th: a +0.0165
  gold gain vanished with a nested teacher; 13th used nested CV).

### 3.5 Validation that tracked the board

6th: a proxy AUC that compares only positive/negative pairs from the same scanner model × field strength,
weighted by label confidence, tracked public LB at Spearman 0.94 (plain macro AUC 0.86). 1st: chose the label set
to score OOF against by correlating 20–30 LB results. 2nd: tuned on 24 gold studies, checked frozen versions on
the other 34. 127th: per-class probe, class AUC = 12 · LB − 5.5 with 0.5 elsewhere. 13th: gold-58 deltas below
~0.01 were not predictive (Spearman 0.23 among the best nine); paired LB submissions drove label decisions.

### 3.6 Ensembling: fixed weights, real diversity

2nd: hand-set weights never searched; tuned weights lost 0.002–0.007 on held-out gold; a look-alike 0.959 member
added nothing. 3rd: fixed group budgets so seed counts cannot decide a family's weight; plain probability means.
6th: logit averaging beat rank blending; inference-only attention sharpening grew with ensemble size. 16th:
pools of correlated own models plateaued at 0.964 public; a different pipeline helped. 383rd: standardised
log-odds 0.943 vs rank average 0.942.

### 3.7 Site habits

New variants beside the training-only bias term: read each report in its original language so translation slips
do not encode the hospital (2nd); switch site-habit cells to the image read (31st, 847 cells); a transductive
field-strength rank remap (207th, +0.002 public). 16th found a training-only site term did not help on one to
three findings.

### 3.8 Runtime

13th projected over 10 h in plain PyTorch and used TensorRT FP16 engines with CPU heads; 1st used TensorRT and
C++ DICOM reading. 1st measured Kaggle disk reads from about 30 MB/s when the platform was busy to over 500 MB/s
when idle, and submitted at a quiet time. 472nd: `"outputs": null` on markdown cells made nbconvert mark a
finished run as ERROR.

### 3.9 The public-recipe band, final numbers

1,591 teams showed public 0.958. On private, 1,297 teams sit at 0.942 (ranks 584–1,880), the score of an
unchanged copy of the 0.958 notebook (478th). The 0.943 band (287 teams) holds most of the small, pre-registered
variations of it: S6fall as a third expert and log-odds averaging (383rd), re-set Raptor weights (472nd, 478th),
and our K1 blends. The entries in this collection that reached 0.944 or more gave an own model real weight
(207th: 0.25 own share, 0.944; 127th: 0.5 share, 0.946).

## 4. Feedback against our record

These extend the post-mortem's M1–M8; each names its evidence and confidence.

1. **Our image reader was under-fed, so D-B-074's lever looked smaller than it was** (confidence medium).
   aquila_v1 read 15 central slices plus the report and our relabel rules (D-B-078) and reached gold-58 0.909
   against 0.899 for the report labels (+0.010). The 2nd's partial-view pilot (44 of 134 slices) gave the same
   misleading "small gain", and its whole-exam read gave +0.043; the 13th's fixed ~95 slices gave 0.928. A
   whole-study or agent-chosen view would probably have put aquila near 0.93–0.94, a gain report-derived rulers
   still could not have seen (M2), but one gold-58 *could* have seen above its ±0.012–0.016 noise.
   *Lesson:* before judging a reader, give it the whole study (or tools to pick slices) and compare protocols on
   gold by log loss and confident errors, not AUC alone.
2. **We measured labels by AUC only.** 1st selected labels by log loss and confident-error counts and calibrated
   them by temperature. Our label gates (L1–L7) ranked by AUC on report-derived references; no log-loss or
   confident-error measure of a label set appears in discoveries.md (its "calibration" entries are post-hoc
   score offsets, D-B-002) (confidence high).
3. **Training on gold-58 was tested once, in a form and on a ruler that could not show the gain** (confidence
   medium). EXP165/EXP173 (10-10; D-B-103, D-B-109) added the 58 studies at ×4 duplication to one fold-0 recipe
   and judged it on report-derived references: −0.006, attributed to duplication and case mix, and the idea was
   not revisited. The top teams used gold differently: the 58 in all-data training at normal weight with their
   expert labels (2nd, 3rd, 10th, 31st), or a short fine-tune merged back with WiSE-FT, which was the 20th's
   largest single step. With 58 studies the honest check is a split (2nd: tune on 24, check on 34) or a fixed
   recipe plus refit (3rd), not a report-label OOF.
4. **OAI: the mapping, not the dataset, decided the outcome.** We cached OAI and built a MOAKS image panel, used
   mostly as a ruler (M2). The first collection read OAI supervision as a dead end; 13th (+0.008 private) and
   10th show it pays when each finding's mapping is validated and mixed with pseudo-labels by reliability.
   *Lesson:* validate an external label's meaning per finding before judging the dataset (confidence medium).
5. **Weak findings needed their own track.** Every top team that reported per-finding numbers singled out
   lateral OA, PF OA and synovitis (plus lateral meniscus) and built specialists, floors or zoom models for them.
   Our process was macro-AUC first with per-target weights on blends (X's public-tuned w_t). A per-finding error
   budget on day one would have pointed the label work at the three findings where reports and images disagree
   most (confidence medium).
6. **Hand-set weights were enough.** 2nd's never-searched weights beat tuned ones on held-out gold by
   0.002–0.007; 472nd and 478th gained only by moving Raptor weights away from the inherited public-tuned values.
   Our time on per-target weight tuning (M5) had little room to pay, as the flat 0.943 of 21 entries already
   showed (confidence high).
7. **Same tools, different target.** The 1st place ran one Codex/Claude agent session per study to make labels
   and draw anatomy boxes, and says Claude Opus 5.5 proposed and ran most pipeline experiments. We ran a larger
   multi-agent system (17+ Codex lanes, dispatch v3.2, an 8-step gate), but pointed it mostly at process,
   verification and release work (M6). The agents were not the limit; what we asked them to do was
   (confidence high).
8. **A likely mechanism for our slow hidden runs** (hypothesis, confidence low). We recorded that same-compute
   hidden runs differed by about 1.5 h (D-B-280) and that U failed on a "slow day" (D-B-282); three of our four
   Oct 22 afternoon/evening submissions failed after about 9.2 h. The 1st measured Kaggle disk reads falling to
   ~30 MB/s when the platform is busy versus over 500 MB/s when idle, which would explain both. We did not log
   I/O time on hidden runs, so this stays unconfirmed. Next time: log per-stage I/O time inside the kernel and
   keep a larger margin for submissions made close to the deadline.
9. **The counterfactual of the post-mortem holds on the final board.** Text-only top-20 entries (6th, 16th)
   confirm that the own model (M4) matters as much as image labels (M1). On the final board the realistic
   scenario B (private 0.948–0.951) maps to ranks 41–98; strong execution (0.952–0.954) to 22–40; minimal fix
   (0.945–0.947) to 99–203.

## 5. Change log

- 1.0.0 (2026-10-28): second collection — 16 write-ups, final verified leaderboard (6,300 teams; our rank 297),
  prelim → final rank map for the first collection, feedback items 1–9.
