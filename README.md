![RSNA Knee Abnormality Detection](docs/figures/banner.png)

# RSNA Knee Abnormality Detection

<p align="center"><b>English</b> · <a href="README.ko.md">한국어</a></p>

| Document | What it holds | Read |
|---|---|---|
| **Final results** | Private score and rank, the final pair, every late submission | [English](docs/reports/final-results.md) · [한국어](docs/reports/final-results.ko.md) |
| **Discoveries** | The findings log: every result, negative and inconclusive ones included | [English](discoveries.md) · [한국어](discoveries.ko.md) |
| **Consolidated report** | The full research state at the pre-result cutoff (Oct 22, 18:15 KST) | [English](docs/reports/project-summary-20261022.md) · [한국어](docs/reports/project-summary-20261022.ko.md) |
| **Experiment catalogue** | Every submission by family, the seven-lens screens, the K1 holdout tests | [English](docs/EXPERIMENTS.md) |
| **Research catalog** | All 390 discovery IDs with dates and review fields | [English](docs/reports/research-catalog-20261022.md) |
| **Dashboard** | KneeOps, the local research dashboard | [English](docs/DASHBOARD.md) |
| **Working files** | Project rules, plan and queue, event log | [agents.md](agents.md) · [plan.md](plan.md) · [handoff.md](handoff.md) |
| **Sources and licences** | External models, data and images | [NOTICE](NOTICE.md) |

Our working record for the Kaggle competition [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) (Jul 30 - Oct 22, 2026): predict, for every knee MRI study, the probability of twelve findings (ligament and meniscus tears, three compartments of osteoarthritis, effusion, synovitis, Baker's cyst, bone contusion, fracture).

| | Score | Rank |
|---|---|---|
| Final private leaderboard | **0.943** | **312 / 6,425 — silver medal** |
| Final public leaderboard | 0.958 | about 638 / 6,390 |
| Private medal cut-offs | gold 0.954 (rank 22), silver 0.943 (rank 321), bronze 0.942 (rank 642) | |
| Top 3 (private) | Team Kangaroo 0.960, MURAD 0.960, DataTech Club 0.959 | 1-3 |

We finished nine places inside the silver cut-off. 311 teams scored above 0.943; 304 teams displayed exactly 0.943, and we were the first of them. On the public board 1,611 teams displayed 0.958, many of them, judging by the public notebooks at 0.958, running the public recipe. Our two final submissions are rank blends of public checkpoints (about 75-80 % of the rank weight) with our own models (20-25 %): the own ConvNeXt-S 2.5D multiple-instance trio "K1" and, in one of them, an older own bundle "O". Both finals scored 0.943 privately, as did 19 other submissions; no submission of ours scored higher. The team account holds 134 submissions between Sep 11 and Oct 22; 37 of them are untracked temporary or teammate uploads without a description and are left out of this record. Of our 97 pipeline submissions, 87 scored, 8 finished without a score and 2 errored.

![leaderboard](docs/figures/leaderboard_final.png)

![knee MRI examples](docs/figures/knee_mri_examples.png)

*Example knee MRI slices for orientation only. These are openly licensed Wikimedia Commons images, **not competition data**; attribution and licences are in [NOTICE.md](NOTICE.md). No competition image, label, report or prediction is published in this repository.*

## The competition at a glance

| | |
|---|---|
| Competition | [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) (Kaggle Research Code Competition) |
| Host | Radiological Society of North America (RSNA), with the RSNA Annual Meeting AI Challenge |
| Dates | Jul 30 - Oct 22, 2026 (final submission 23:59 UTC); entry and team-merger deadline Oct 15 |
| Participation | 6,425 teams, 7,683 participants, 161,275 submissions |
| Prizes | $77,000: main leaderboard 1st-10th ($9,000 to $5,000) and an Efficiency Track (1st-3rd, $7,000 / $6,000 / $5,000) |
| Data | Knee MRI studies (multi-series DICOM) for twelve abnormalities; the first RSNA AI Challenge dataset that pairs every imaging study with its original radiology report |
| Targets | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture |
| Metric | Macro AUC: the mean of the twelve per-target ROC AUCs |
| Submission | `submission.csv` with `StudyInstanceUID` and one confidence column per target |
| Code requirements | Kaggle Notebooks only; CPU or GPU notebook <= 9 h run time; internet disabled; freely and publicly available external data and pre-trained models allowed |
| Efficiency Track | Ranks selected submissions by a score that combines private AUC with runtime (seconds / 32,400); our final runs took about 6.8-8 h, so we did not target it |
| Data licence | Competition data under the Kaggle competition rules and the RSNA MIRA licence; rule 4.b forbids redistribution, so this repository contains no competition images, reports, labels or study IDs |

Winners must also publish a short video, their open-sourced code and weights, and a public model, in addition to the standard Kaggle winner obligations. We are not prize winners; this repository is our own record.

**Acknowledgements.** The challenge was organised by an RSNA team led by Po-Hao "Howard" Chen and Naveen Subhas (Cleveland Clinic), with de-identified MRI studies and reports contributed by institutions on five continents, annotations by members of the Society of Skeletal Radiology and the International Skeletal Society, report and image QC reviewers, and annotation tooling from MD.ai. The full list is on the [competition overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview).

**Citation** (as given by the host):

> Po-Hao "Howard" Chen, Naveen Subhas, Robyn Ball, Pieter Baeyens, Errol Colak, Ali Emami, Hillary Garner, Jacob Kazam, Hui-Ming Lin, Luciano Prevedello, Daniel Schneider, Jason Sho, Ryan Holbrook, and María Cruz. RSNA Knee Abnormality Detection. https://www.kaggle.com/competitions/rsna-knee-abnormality-detection, 2026. Kaggle.

## Contents

1. [The task and the metric](#1-the-task-and-the-metric)
2. [What we built on, and what we added](#2-what-we-built-on-and-what-we-added)
3. [Labels: sources and relabelling](#3-labels-sources-and-relabelling)
4. [Rulers, gates and the seven-lens readout (L1-L7)](#4-rulers-gates-and-the-seven-lens-readout-l1-l7)
5. [How the ensemble was decided](#5-how-the-ensemble-was-decided)
6. [The final submissions](#6-the-final-submissions)
7. [Mechanisms in detail](#7-mechanisms-in-detail)
8. [How the work evolved](#8-how-the-work-evolved)
9. [Validation](#9-validation)
10. [Final week, failures and runtime](#10-final-week-failures-and-runtime)
11. [Key observations](#11-key-observations)
12. [Top-team strategies](#12-top-team-strategies)
13. [Self-feedback](#13-self-feedback)
14. [How we used AI agents and skills](#14-how-we-used-ai-agents-and-skills)
15. [KneeOps local dashboard](#15-kneeops-local-dashboard)
16. [Repository layout](#16-repository-layout)
- [Reports and navigation](#reports-and-navigation)

Discovery IDs such as `D-B-264` refer to entries in [`discoveries.md`](discoveries.md) (Korean translation: [`discoveries.ko.md`](discoveries.ko.md)), the shared findings log kept during the competition. Every submission with its family and scores is in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) and [docs/data/submissions.csv](docs/data/submissions.csv). Dates are UTC (Kaggle submission time) unless marked KST. "Display" means the score as Kaggle shows it, rounded to 0.001.

## 1. The task and the metric

- **Input:** one study = several DICOM series of one knee (sagittal, coronal and axial planes; fluid-sensitive or not, fat-suppressed or not), typically 20-45 slices per series, with a long tail to a few hundred. Transfer syntaxes include JPEG Lossless and JPEG 2000. About 4,407 training studies from many sites, scanners and protocols.
- **Labels:** only **58 training studies carry official per-finding labels** ("Gold58"). For the others the free-text radiology report is provided (several languages) and participants derive labels themselves. The hidden labels are image reads under written severity rules (for example, MCL = acute high-grade partial or complete tear; low-grade sprains and chronic changes are negative; effusion and Baker's cyst count only when moderate or large).
- **Test:** about 1,300 hidden studies, no reports. The public board uses roughly 30 % (about 390 studies), the private board the rest (about 910).
- **Metric:** mean of the 12 per-finding ROC AUCs (macro AUC). Only the ranking within each finding matters, which is why every blend below works on ranks.
- **Runtime:** code competition, internet off, 9-hour notebook limit for the whole hidden set. We used Kaggle's 2 x T4 GPU sessions.

## 2. What we built on, and what we added

![pipeline overview](docs/figures/pipeline_overview.png)

**Public work we built on** (credits and licences in [NOTICE.md](NOTICE.md)):

- nartaa's all-data CoAtNet "Raptor" SWA checkpoints at 384 px and 224 px (R384, R224; fine-tuned with OAI images);
- heliosli's compact ConvNeXt-S 2.5D bundles (S6, then R7, finally R6 `fe055406`) and their serving code from heliosli's OrthoFoundation code dataset and public notebook;
- goodpjw2008's public 0.958 per-target blend weights, R224 share and flip TTA (the same recipe appears in evgendvorkin's versia-7);
- dreaddevelopment's Raptor training recipe and report labels ("Dread"), riadmohamed42's JEV labels and other public label sets;
- jiweiliu's and mattiaangeli's 0.943 inference stack (our reference point in early October) and mattiaangeli's DINOv3 fold weights (the A5 member tested on Oct 21).

**External models and sources** (Kaggle pages checked against Kaggle's public listing on Oct 23; the full list, including every label set and the inputs of the copied public stacks, is in [NOTICE.md](NOTICE.md)):

| Source (author) | Kaggle page | Used for |
|---|---|---|
| Raptor CoAtNet SWA checkpoints R384, R224 (nartaa) | [dataset](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007) | R in both finals |
| Compact ConvNeXt-S 2.5D bundle R6 `fe055406` (heliosli) | [dataset](https://www.kaggle.com/datasets/heliosli/rsna-knee-r6-frozen-fe055406); bytes from the versioned copy by [rostigor](https://www.kaggle.com/datasets/rostigor/rsna-knee-r6-fe055406) | R6 in both finals (private copy) |
| OrthoFoundation code dataset (heliosli) | [dataset](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004) | R6 serving code; K1's starting trainer |
| Earlier compact bundles R7 and S6 (heliosli) | [R7](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss), [S6](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-v1) | compact member Oct 11-16 |
| 0.957 one-click notebook (heliosli) | [notebook](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click) | S-branch serving sources |
| Public 0.958 notebook (goodpjw2008) | [notebook](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958) | w_t, R224 share, flip TTA |
| 2.5D ConvNeXt reader (goodpjw2008) | [dataset](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader) | overlay test (H-B-190) |
| RSNA_Versia_7 (evgendvorkin) | [notebook](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7) | read only |
| 0.943 recipe and DINOv3-S fold weights (mattiaangeli) | [notebook](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-speedy-raptors-the-original), [dataset](https://www.kaggle.com/datasets/mattiaangeli/knee-mri-fold-weights) | reference recipe; A5 member |
| 0.943 fast 2xT4 inference (jiweiliu) | [notebook](https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference) | our Oct 2 copy |
| 0.941 stack, preset probe22 (maverickss26) | [profile](https://www.kaggle.com/maverickss26) (exact page not verified) | our Sep 11-12 start |
| Raptor training recipe and Dread labels (dreaddevelopment) | [notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model), [dataset](https://www.kaggle.com/datasets/dreaddevelopment/rsna-knee-labels) | own CoAtNet recipe; Dread ruler; 75 % of L2 |
| JEV labels (riadmohamed42) | [dataset](https://www.kaggle.com/datasets/riadmohamed42/jev-knee-labels) | JEV hybrid ruler |
| Report labels (flight0234, stevenleehans, pilkwang) | [flight](https://www.kaggle.com/datasets/flight0234/rsna-knee-hybrid-report-labels), [steven](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels), [pilkwang](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels) | consensus ruler |
| 11-label evidence fusion (omerzalman) | [dataset](https://www.kaggle.com/datasets/omerzalman/rsna-knee-11-label-evidence-fusion) | extra lens (D-B-259) |
| RadImageNet ResNet-50 mirror (marwanmath) | [dataset](https://www.kaggle.com/datasets/marwanmath/resnet-50-radimagenet-marwan) | init of our ResNet-50 partners |

Outside Kaggle: [OAI](https://nda.nih.gov/oai) (through nartaa's checkpoints and our OAI panel), [RadImageNet](https://www.radimagenet.com/), [timm](https://github.com/huggingface/pytorch-image-models), [PyTorch](https://pytorch.org/).

**What we added:**

| Area | What | Where it ended up |
|---|---|---|
| Labels | Own image-and-report labels **Aquila** (aquila_v1, 4,405 studies read by Codex "astra" agents), definition-aligned relabel_v1, consensus filters, and the **L2** training labels (0.75 Dread + 0.25 Aquila) | trained every own model from Oct 10 (D-B-095) |
| Own models | CoAtNet EXP143v2 (OAI-free) and EXP155 (OAI-masked targets), partner bundle Q (RadImageNet ResNet-50, ConvNeXt-T, ConvNeXt-S MIL), and **K1**: own ConvNeXt-S 2.5D multiple-instance (MIL) model, three checkpoints | O in final A; K1 in both finals |
| Serving | shared decode and shared render (K1 classifies the windows the R6 member already rendered, cutting K1 from about 5.2 h to under 0.5 h per checkpoint), offline JPEG Lossless / JPEG-LS decoders, per-study fallback, fail-loud runtime | both finals |
| Evaluation | four report-derived rulers, the seven-lens readout, a fixed 977-study holdout, an evidence ledger, an image-read OAI panel | every decision after Oct 9 |
| Process | independent build/review lanes, one-attempt submission ledger, 8-step submission gate, pre-committed final-pair rules | every late submission |

## 3. Labels: sources and relabelling

### 3.1 The problem

Only Gold58 has official labels. Every other training label is read from a report, and reports differ from the official image reads in systematic ways: they are often silent about synovitis and small effusions, may call a sprain a tear, or mention old injuries that the rules grade negative. On Gold58 the public report-derived label sets plateau near 0.90 macro AUC and averaging several of them does not help (0.895-0.896, below the best single set; D-B-014). When one set is wrong on Gold58, another is wrong 48-100 % of the time against a 17-21 % base rate (D-B-040), so more report-derived sets add little independent information. Eleven of 24 public label files copy the 58 gold rows exactly and cannot be scored on them (D-B-014). The host confirmed that labelling the training reports with a commercial LLM is allowed, provided the report text is not redistributed (D-B-029).

### 3.2 Label sources

| Source | Origin | How we used it |
|---|---|---|
| **Dread** | dreaddevelopment's public report labels (4,349 rows, Gold58 excluded) | training labels for early own models; ruler |
| **JEV hybrid** | riadmohamed42's public labels | ruler; best Gold58 agreement among public sets (0.9037) but tuned on 74 studies that include Gold58, so optimistic (D-B-068) |
| flight_v4, steven_v4, pilkwang_v2 | public report-derived sets | consensus members; per-target ruler map (D-B-112) |
| relabel_v1 | our definition-aligned LLM report labels (prompt v1.1) | trained one public test: 0.913 vs 0.914 on Dread, inconclusive (D-B-038) |
| **Aquila (aquila_v1)** | our own labels (EXP148, Oct 9): a Codex astra reader given per-plane image montages, the report text, a report-derived prior label and the official rules, up to 12 parallel calls, about 5 h of reading; 4,405 studies, 0 missing or malformed, Gold58 removed, clipped to 0.02-0.98 (4,347 rows) | 25 % of the L2 training labels; ruler |
| **Consensus** | agreement of independent report-derived sets (for example relabel_v1, steven_v4, pilkwang_v2); on Gold58 three such sets agree on 80 % of study-finding pairs, and where they agree the label matches Gold58 89.6 % of the time vs 81.7 % for a single set (D-B-035) | ruler |
| **L2** | 0.75 Dread + 0.25 Aquila soft labels | training labels for every own model from Oct 10, including K1 |
| Gold58 | the 58 official labels | diagnostic only, never a training target or epoch selector; exposed by repeated use |

Why Aquila was built, and what it did:

- **Image evidence helps the reader.** On 20 Gold58 studies a vision LLM given three per-plane montages (15 central slices each), the report-derived label and the official rules beat the report label alone; the largest gain was Effusion (0.797 -> 0.891-0.974). One reader gained +0.0148 macro, replicated on 38 fresh Gold58 studies (D-B-074). Adding the report text helped the astra and Opus readers slightly (0.919 / 0.918) and confused a third (0.893) (D-B-078). The astra variant with report text ("variant F") was run on the whole training set.
- **Diagnosed misses.** On Gold58 the strict verbal rule made astra miss 7 of 9 MCL positives and 8 of 21 PF OA positives (rank order intact, probabilities deflated), and 12 of 27 Synovitis positives where the report is silent. This is why Aquila is mixed in as 25 %, not used alone.
- **Full-scale effect.** L2 labels raised five-fold JEV by +0.0050 for ResNet-50 (two seeds, pooled over 4,343 studies), +0.0054 for ConvNeXt-T and +0.0058 for DINO-base (D-B-095, D-B-098, D-B-111). Removing the Aquila share lost 0.0069 on fold 0 with every target down; 50 % was flat (D-B-123); 60 % on five targets hurt the report rulers and Gold58 and helped only the OAI image panel (D-B-166).
- **Ruler quality.** In the retrospective ruler check, a binarised Aquila ordered 102 of 118 submission pairs the way the public board did, tied with consensus and ahead of JEV (99), Dread (96) and flight (93) (D-B-081).

### 3.3 Relabelling and label-mixing attempts

| ID | Attempt | Result |
|---|---|---|
| D-A-006 | Blinded audit of disputed ("silver") labels | insufficient direct evidence to auto-correct Dread |
| MCL definition audit (Oct 3) | MCL positives in 4,345 studies: Dread 214, Qwen 648, flight 660, pilkwang 749, against the official "acute high-grade" definition | majority vote or wholesale swap would train low-grade and old injuries as positive; Dread kept |
| D-A-028 | Blinded review of 120 MCL reports | all 55 negative candidates were already Dread negatives; no correction |
| D-A-021 | Report labels mixed with teacher predictions (R2) | weak-CV +0.016, not independent: teacher ancestry leaks across folds |
| D-A-022 | Report labels + image model on Gold58 | complementary (+0.029), exploratory; motivated image-informed labels |
| D-B-011 | Leak-free nested teacher-label mix (0.5 Dread + 0.5 inner-OOF teacher) | -0.0058 vs matched control: helps where the teacher is strong (PF OA +0.014), hurts MCL (-0.050) |
| D-B-027, D-B-038 | Distilled labels; definition-aligned relabel_v1 | public 0.912 and 0.913 vs 0.914 |
| D-B-084 | Consensus-weighted supervision | seed-inconsistent (+0.0019 / -0.0004), stopped |
| D-B-103, D-B-109 | Oversampling Gold58 x4 | -0.0058, explained by study duplication, not label value |
| D-B-115, D-B-119, D-B-124, D-B-130 | Filling report-silent Synovitis from Effusion, silent PF OA from Aquila | Synovitis -0.0033 with ACL/MCL harm; PF OA gains not reproduced |
| D-B-180, D-B-187, D-B-191 | Soft pseudo-labels / distillation from strong public teachers | NO_GO; public 0.944 vs 0.946; 0.948 vs 0.949 |
| D-B-235 | Blinded expert MRI adjudication (96-study pilot) | roster and blank forms prepared, no readers secured; not run |
| D-B-259 | omerzalman 11-label evidence-fusion set as an extra lens | ordered the two test pairs no better than our rulers (r 0.97-0.99 with them) |
| D-B-267 | K1 out-of-fold failure map on 4,167 studies | 87 % of cells have agreeing labels; most ranking errors sit on agreed cells (FN 260 / FP 1,227); largest pockets look like co-finding shortcuts (Lateral OA without lateral-meniscus abnormality AUC 0.887 vs 0.956) |

## 4. Rulers, gates and the seven-lens readout (L1-L7)

![Label sources feed four report-derived rulers, which score candidates on the 977-study fold-0 holdout against preregistered gates](docs/figures/labels_and_rulers.svg)

*Label sources, the four rulers and the held-out cohort behind every K1 gate; Gold58 stayed a diagnostic because repeated use exposed it.*

### 4.1 The four rulers

Every local comparison was scored against four report-derived label sets: **JEV hybrid, Aquila, Dread, Consensus**. Each gives a macro-12 AUC difference between a candidate and its matched control on the same studies, with a paired bootstrap over studies (300-2,000 draws). They are correlated (all report-derived), so agreement among them is weaker evidence than four independent tests. A per-target ruler map built from Gold58 agreement rated the rulers adequate (HIGH) only for ACL and Medial OA, MEDIUM for eight findings and LOW for Synovitis and Contusion, where no report-derived ruler is adequate (D-B-112).

The **weak four** are Lateral Meniscus, PF OA, Lateral OA and Synovitis: the findings with the largest deficits of our models (MCL, PF OA, Lateral Meniscus and Lateral OA explained 47.6 % of an early model's weak-label deficit, D-A-026) and the ones where the public stack was weakest. From Oct 11 every protocol preregistered the mean change over the weak four on mean(JEV, Consensus) as its primary number, with harm diagnostics: macro-12 >= -0.001, no other target below -0.010, Dread macro >= -0.003.

### 4.2 The seven lenses

A single gate threw away small real gains and passed noise, so from Oct 11 (Taeyang's instruction, recorded in `agents.md`) each screen was read through seven lenses and given a holistic label.

| Lens | Rule | Why it exists | Introduced / corrected |
|---|---|---|---|
| **L1** weak four | mean change over the weak four >= +0.001 on mean(JEV, Consensus) | the targets where gains were most needed | weak-four readout, Oct 11 01:07 KST |
| **L2** macro-12 | macro-12 change >= +0.001 on mean(JEV, Consensus) | overall metric | advance rule, Oct 11 22:1x KST |
| **L3** single ruler | any one ruler >= +0.002 macro, every other ruler >= -0.001 | catch a gain one ruler sees clearly without harm elsewhere | Oct 11 |
| **L4** per target | a target >= +0.003 on at least 3 of 4 rulers; no mean gate | per-finding routing can use a model for the targets it improves without replacing it | Oct 11; corrected Oct 12 (D-B-158): the first tool combined a mean >= +0.003 with "positive on 3 rulers" and passed a synthetic case where only one ruler reached +0.003; `tools/multi_lens_readout_l4_v2.py` implements the registered rule |
| **L5** seeds | seed average and sign agreement over **every** existing same-recipe seed pair | one same-machine seed change moved macro-12 by -0.006 (D-B-017), as large as most effects | Oct 11; "every pair" Oct 12 23:2x KST |
| **L6** bundle | replacing the counterpart in the partner bundle on the same rows changes the bundle by >= +0.0005 | own models were used as blend partners, and a standalone or complement gain can vanish in a blend (a +0.0076 OOF complement gain transferred as +0.001 public, D-B-057) | Oct 11 |
| **L7** image evidence | Gold58 and/or the OAI MOAKS image-read panel (557 OAI knees; weak four mapped as Lateral Meniscus, Lateral OA, PF OA, Synovitis) | report rulers cannot measure Synovitis or Contusion; image evidence is decisive for Synovitis and PF OA | Oct 11; OAI panel from Oct 12 (D-B-135) |

![Seven lenses L1-L7 with their thresholds feed one holistic decision: ADVANCE, PER-TARGET, HOLD or DROP; an unexplained missing lens forces HOLD](docs/figures/seven_lens_readout.svg)

*The seven-lens readout: each lens is run or excused in writing, then one weighed judgement gives the label.*

**Decision labels.** ADVANCE (enough lenses agree for a whole-model next step), PER-TARGET (use only for the targets L4 supports), HOLD (unclear or incomplete), DROP (lenses agree on harm). The label is a weighed judgement of agreement count, size, seed and ruler consistency, not a vote and not a single-lens veto. From Oct 12 23:2x KST every lens had to be made **eligible** by running the needed inference (L6 partner predictions on the same rows, L7 panel predictions), or carry a written reason (for example, an OAI-trained arm cannot be scored on the OAI panel; Gold58 is exposed). **An unexplained missing lens forces HOLD**, never ADVANCE or DROP. Where the lenses disagree on added false positives, the public reading arbitrates. The registered gate of each experiment was still recorded beside the holistic label; a failed gate was not relabelled a pass.

**Retro scan (D-B-129).** Re-reading 74 past contrasts from 27 screens with L1-L7 (`artifacts/experiments/MULTI_LENS_RETRO`, 45 tests) flipped 16 old HOLD/reject verdicts to ADVANCE or PER-TARGET. The actionable one was ConvNeXt-S (weak four +0.0066, macro +0.0013; only MCL -0.012 with CI to +0.002), which led to the ConvNeXt-S MIL partner (H-B-105). Single-seed per-target flips were treated as hypotheses because L4 makes 12 comparisons per arm.

### 4.3 Worked examples

**Coverage-preserving sampler for the own CoAtNet (D-B-156, Oct 12).** Two seeds, 888 fold-0 studies, mean of plain and mirror views.

| Lens | Reading | Pass? |
|---|---|---|
| L1 weak four | +0.0008 [-0.0016, +0.0036] | no |
| L2 macro-12 | -0.0003 [-0.0022, +0.0017] | no |
| L3 single ruler | +0.0007 | no |
| L4 per target | ACL +0.0115 [+0.0061, +0.0181], all 4 rulers, both seeds (after the D-B-158 correction; PF OA had wrongly passed) | ACL only |
| L5 seeds | -0.0004, directions disagree | no |
| L6 bundle (50:50 with Q7) | -0.0003 [-0.0013, +0.0008] | no |
| L7 OAI panel | weak four +0.0103 (seed 42) / +0.0073 (seed 43), CIs include 0 | positive, uncertain |

Label: **PER-TARGET (ACL)**. The automated screen had first read "no adoption" with only L1-L4 eligible; the owner's amendment made the label HOLD until L5-L7 were run, which is how the rule "an unexplained missing lens forces HOLD" arose.

**Multi-view inputs: full field + compartment crops as extra MIL instances (D-B-198, Oct 15-16).** Fold 2, seeds 42/43. Registered gate: NO_PASS (four-ruler macro +0.0012 < +0.002; seeds -0.0025 / +0.0033; Baker's -0.0074 below the -0.005 floor).

| Lens | Reading | Pass? |
|---|---|---|
| L1 | +0.0006 [-0.0045, +0.0061] | no |
| L2 | +0.0004 [-0.0027, +0.0037] | no |
| L3 | +0.0010 | no |
| L4 | Medial OA, Lateral OA, Contusion (+0.0074) | yes |
| L5 | +0.0011 [-0.0021, +0.0044] | weak |
| L6 | +0.0015 [+0.0005, +0.0023] | yes |
| L7 | ineligible with reason (Gold58 exposed, no OAI crop cache); Gold58 descriptive +0.0051 | - |

Label: **ADVANCE to a fold-3 confirmation**, not to release. Fold 3 (seeds 44/45): four-ruler macro -0.0017 with all rulers negative, OA reversed, Fracture -0.022, L6 +0.0006 -> HOLD; combined folds 2+3 showed no support.

**OA-only crop routing (D-B-197, Oct 15).** Fold 3, seeds 44/45: route only Medial/Lateral OA to the crop expert. Registered gate FAIL (Medial OA -0.0071 on 0/4 rulers). Lenses: L1 +0.0018 pass, L2 -0.00002, L3 +0.0005, L4 Lateral OA +0.0079 on 4/4 rulers pass, L5 +0.0001 pass, L6 +0.0003 [-0.0001, +0.0008], L7 ineligible. Label: **DEPRIORITISED, not discarded** (Lateral OA replicated on two folds). The same Lateral OA expert was retested inside K1 on Oct 22: +0.0022 on Lateral OA, below its +0.006 gate (D-B-271).

### 4.4 Preregistered K1 gates (Oct 18-22)

For K1 the readout was simplified to one fixed set: K1 trained on folds 1-3 (3,190 studies), evaluated on the **977-study fold-0 holdout** (976 in the primary cohort), the four rulers, and a rule written to a file before the run. Typical rule: candidate >= control on at least 3 of 4 rulers AND mean macro change >= +0.002 (+0.001 for an additional ensemble member, +0.003 for TTA, +0.006 on the single target of an expert), no target below -0.005. A different agent re-computed every result. In the last 48 hours one of seven K1 changes passed (full-bag 4th member, D-B-278) and six failed (D-B-264, 268, 269, 271, 272, 273, 279); details in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#local-holdout-experiments-k1-977-study-fold-0).

### 4.5 Why the rulers over-credit own-model changes

Our own models are trained on report-derived labels, and the rulers are report-derived, so every ruler (and Gold58) rewards a larger own-model share. The public board disagreed each time it was tested:

| Change | Local evidence | Public result |
|---|---|---|
| Own share 0.15 -> 0.25 (D-B-246) | Gold58 +0.0005 | 0.958 -> 0.957 |
| Drop R224 (D-B-246) | Gold58 +0.0011 | 0.958, ranked below its parent by auto-selection |
| Per-target K1 share vector up to 0.5 (av2, D-B-251) | chosen on the 976 holdout | 0.958 -> 0.957 |
| Same pair on the 11-label fusion lens (D-B-259) | 0 of 7 lenses ordered it as the board did | - |
| Full-bag K1 as a 4th member (D-B-278 -> P4, D-B-282) | +0.00145 K1-only, 4/4 rulers | 0.958, below p25 by auto-selection; private 0.943, the same as p25 |

From Oct 21 own-model weights were changed only on public evidence, and K1 was improved at a fixed weight.

## 5. How the ensemble was decided

All blends are per-target rank averages with shares fixed before submission (mechanics in [7.1](#71-rank-averaging-with-average-ties)).

![The final formula built layer by layer: public P with per-target weights, O at 0.15, the R224 share, K1 at 0.20, the p25 variant, the flat-P hedge and the final pair X + F](docs/figures/ensemble_decisions.svg)

*How each layer of the final blends was fixed, with its public reading and discovery ID; both finals scored 0.943 privately.*

The steps, in order:

| Date | Step | Evidence used | Public | Decision |
|---|---|---|---|---|
| Oct 7 | Public 0.943 stack 80 % + own CoAtNet/ResNet 20 % (H-B-15) | five-fold OOF | 0.942 vs 0.943 | no gain; own models too weak (D-B-031) |
| Oct 9-10 | nartaa's OAI CoAtNet 0.949 + own CoAtNets, 1/3 each (blend3) | own Gold58 0.935, OOF | 0.950-0.951 | rank blending with fixed shares |
| Oct 11-12 | 0.85 public part + 0.15 own part (A5q, C85); P956 = 0.5 R + 0.5 heliosli S6 | 0.15 carried over from the registered partner share (D-B-128: 0.45 looked better on fold 0, the nested fallback kept 0.15) | 0.954-0.955 | own share fixed at 0.15 |
| Oct 13 | S6 -> heliosli's R7 bundle (C85-R7) | release parity | 0.957 | adopt |
| Oct 14-15 | R slot = 0.65 r(R384) + 0.35 r(R224), share from the public recipe (H-B-164); R7 full flip TTA | auto-selection order (D-B-195) | 0.957 | adopt both |
| Oct 16 | R7 -> R6 `fe055406` with full flip (H-B-174) | Gold58 +0.0033 [-0.0002, +0.0073] (D-B-201); not replicated on 461 OAI knees (D-B-208) | 0.957, auto-selected | adopt |
| Oct 17 | Public per-target P weights w_t (H-B-189) | public recipe; Gold58 put them near 0 (D-B-212) | **0.958**, auto-selected first | adopt, flagged as public-tuned |
| Oct 18 | Own-side changes inside O (INF2, INF3 per-checkpoint Q pooling, C4) | exposed Gold58 | 0.958 ties | INF3 kept as parent |
| Oct 19 | Consult (fable + astra): INF3 : K1 = 80 : 20 | Gold58 K1 at 0.20 +0.0006 | - | K1 share 0.20 |
| Oct 20 | In-pass K1, 1 -> 3 checkpoints (x3-v2) | 976 holdout: 2-checkpoint average +0.0015..+0.0029 on all rulers (D-B-245) | 0.958, auto-selected (D-B-249) | **final A** |
| Oct 20 | Drop O, K1 share 0.25, K1 stride 2 (p25) | stride 2 > 3 on 4/4 rulers, x3 +0.0007..+0.0017 (D-B-247) | 0.958, auto-selected with x3-v2 (D-B-260) | O not needed for the top display |
| Oct 20-21 | Per-target K1 share vector (av2); fold/all-data K1 members (C) | 976 holdout | 0.957, 0.957 | rejected (D-B-251, D-B-263) |
| Oct 21 | Flat P (all w_t = 0.5) on the p25 recipe (P25F) | hedge against public-tuned w_t; consult estimate of the private effect 0 +/- 0.0015 | 0.957 | **final B**, pre-committed (D-B-274) |
| Oct 21-22 | mattiaangeli's DINOv3-S folds (A5) at 0.15 inside P; A5 + flat P (DF) | provenance-clean new family | 0.956, 0.955 | closed by the pre-committed rule (D-B-276, D-B-281) |
| Oct 22 | Full-bag 4th K1 member (P4); union of both families (U, U4); flat twin (F4) | holdout GO (D-B-278); consult P(>= 0.959) about 3-8 % for U | P4 0.958 (not selected); U, U4, F4 no score | pair unchanged (D-B-282) |

**Consults.** At each scheduling point (Oct 19, 20 18:35, 21 16:57, 22 02:46, 12:2x, 18:0x and 23 00:3x KST) the same brief went to two independent models: fable (a Claude model) and an astra Codex lane. Their estimates were recorded with the rule; for example, on Oct 21 P(public 0.959) 10-20 % and P(0.960) 1-5 %, and on Oct 22 "no lever with P(>= 0.959) >= 10 %" from three Claude readers and one astra reader of the whole findings log (D-B-277).

**Pre-committed final-pair rules.** From Oct 20 the rule for every remaining slot was written into `plan.md` (Final 10-slot schedule, Updates 1-12) before the result it depended on. The default pair was X (x3-v2: best auto-selected, contains O and the public-tuned w_t) + F (P25F: no O, flat P weights, so it differs from X in two structural ways). Kaggle's auto-selection would have kept x3-v2 + p25, which share the public-tuned weights; F replaced p25 manually on Oct 23 00:3x KST, with both consultants agreeing. A switch was allowed only for a candidate scoring >= 0.959 by Oct 23 08:00 KST; display differences below 0.002 were treated as ties. None qualified. Private: X 0.943, F 0.943, p25 0.943.

## 6. The final submissions

### 6.1 Members

| Symbol | Member | Source | Share |
|---|---|---|---|
| R | R384 and R224 all-data CoAtNet (coatnet_rmlp_2_rw) SWA checkpoints; R = r(0.65 r(R384) + 0.35 r(R224)) | nartaa, public dataset | w_t inside P |
| R6 | ConvNeXt-S compact 2.5D bundle `fe055406` with fluid/state stackers and side classifier, full flip TTA | heliosli (served from a private copy); serving code from heliosli's OrthoFoundation code dataset | 1 - w_t inside P |
| P | P_t = r(w_t r(R)_t + (1 - w_t) r(R6)_t) | goodpjw2008's public weights, or flat | - |
| O | own bundle: EXP143v2 (own CoAtNet, OAI-free, L2 labels), EXP155 (own CoAtNet with OAI-masked targets), Q partners (RadImageNet ResNet-50, ConvNeXt-T, ConvNeXt-S MIL) pooled per checkpoint with inverse class-cost weights | own | 0.15 inside INF3 |
| K1 | own ConvNeXt-S 2.5D MIL; per-target rank mean of three checkpoints (v2, v3-G seed 0, v3-G seed 1); FP16 | own | 0.20 or 0.25 |

**K1 in detail.** Each series is cut into windows of three slices at 3.5 mm physical spacing, rendered at 384 px and cropped to 352 px, with anatomy anchors; every plane and contrast contributes windows. A ConvNeXt-S encoder embeds each window; class-specific attention pools all windows of the study into one vector per finding; the loss is a study-level weighted BCE on L2 soft labels. Training: folds 1-3 (3,190 studies), 10 epochs, 15,950 updates, EMA, at most 64 windows sampled per study per step (sampling that covers every series, candidate stride 3). It started from heliosli's public compact trainer. v1 collapsed to the constant-prediction loss at epoch 3 (D-B-214); v2 lowered the learning rates to 1e-4 (encoder) / 3e-4 (head) with a 1-epoch warm-up (D-B-218); v3-G fixed the parameter groups (D-B-220); v3-G seed 1 completed the trio (D-B-245). On the holdout K1 v2 was as strong as heliosli's R7 fold-0 model (macro within -0.005 on all four rulers) and much less correlated with it (mean per-target Spearman 0.777), which is why it was used as a blend member. Standalone on Kaggle it scored 0.944 public / 0.931 private (K2, D-B-219).

### 6.2 Formulas

| Name | Formula | Public | Private |
|---|---|---|---|
| INF3 | r(0.85 r(P) + 0.15 r(O)) | 0.958 | 0.943 |
| **X = x3-v2** (`knee-lb-r6w958k1x3sr20-v2`) | r(0.80 r(INF3) + 0.20 r(K1, stride 3)), runtime v5 | 0.958 | 0.943 |
| p25 | r(0.75 r(P) + 0.25 r(K1, stride 2)), no O, runtime v7 | 0.958 | 0.943 |
| **F = P25F** (`knee-lb-r6w958k1n3s2p25flatpv10-v1`) | p25 with flat P (all w_t = 0.5), runtime v10 | 0.957 | 0.943 |
| P4 / F4 | p25 / P25F with a 4th K1 member (full-bag fold-0 checkpoint) | 0.958 / - | 0.943 / - |
| U | r(0.5 r(X_u) + 0.5 r(Y_u)), X_u = r(0.8 r(INF3) + 0.2 r(K1)), Y_u = r(0.75 r(P) + 0.25 r(K1)), K1 stride 2 | - | - |
| A5 / DF | p25 with P' = r(0.85 r(P) + 0.15 r(A5)); DF also flat P | 0.956 / 0.955 | 0.941 / 0.940 |

## 7. Mechanisms in detail

### 7.1 Rank averaging with average ties

For each finding t, a member's predictions over the **full test roster** (all hidden studies at once, not a subset) are turned into percentile ranks; tied predictions get the average of the ranks they span. Ranks are combined with the fixed shares and ranked again, so every level of the formula is a rank and the output for each finding is a rank in (0, 1]. Because the metric is a per-finding AUC, only order matters, and ranking removes calibration differences between members (a CoAtNet's probabilities and a stacker's outputs live on different scales). Member probabilities are kept as float32 before ranking. A study on which K1 failed takes the parent's rank in the K1 slot before the outer rank, so it is neither dropped nor pushed to an extreme.

### 7.2 Per-target public weights w_t

w_t is the Raptor (R) share inside P for finding t. The public 0.958 recipe sets Lateral OA 0.2, PF OA 0.6, Synovitis 0.8, Baker's 0.35 and 0.5 for the other eight findings. These four values were tuned by their author on the public board, so part of their 0.958 may be selection on the 390 public studies (winner's curse). Applying them to our R6 line raised it from 0.957 to 0.958 (H-B-189, D-B-213); Gold58 put their effect near zero (D-B-212). Flat P (all 0.5) cost one display step on public (0.958 -> 0.957, D-B-274) and changed nothing privately (P25F 0.943 = p25 0.943).

### 7.3 R224 share inside the Raptor slot

R is itself a blend of two checkpoints from nartaa's dataset: R = r(0.65 r(R384) + 0.35 r(R224)). The 0.35 comes from the public recipe. Adding R224 (H-B-164) displayed 0.957 like its parent, but auto-selection placed it above the other 0.957s (D-B-195). Dropping R224 later (H218p) displayed 0.958 but was ranked below the parent by auto-selection (D-B-246) and scored 0.942 privately, the only 0.958 of ours below 0.943.

### 7.4 Eval stride

At inference a series is covered by windows whose centres are spaced by the eval stride in slices. K1 at stride 2 instead of 3 uses about 1.44 x the windows and was better for every checkpoint and the trio on all four rulers (trio +0.0007..+0.0017, P(>0) >= 0.95; D-B-247). Stride 1 changed nothing (-0.00011 / +0.00013 / +0.00004 / +0.00003; Spearman 0.9996; D-B-268), so window density saturates at stride 2. For R6, stride 2 instead of 1 cut R6 inference by 35 % on 32 studies with per-target Spearman 0.983-0.999 (D-B-250), but its public reading was at best neutral (S1 0.958, not auto-selected; D-B-262), so the finals keep R6 at stride 1. X keeps K1 at stride 3; stride 2 entered with the p25 family.

### 7.5 Flip and TTA choices

| Choice | Evidence | Used? |
|---|---|---|
| R7 full flip TTA | +0.0013 [+0.0003, +0.0023] on 989 exposed held-out studies (D-B-181); about 0 on 461 OAI knees (D-B-209) | yes, as R6 full flip |
| R6 flip vs no flip | +0.00061 on OAI 461, P(>0) about 0.76 (D-B-209) | kept as an uncertain gain |
| R224 anatomical mirror | public 0.957 vs parent 0.958 (D-B-217) | no |
| Plane-aware flip TTA on own ResNet-50 | five-fold Dread +0.0021 (D-B-056) | own R50 models |
| Serving-time scale TTA (320 crop + 384 full field) | -0.00004 macro (D-B-189) | no |
| K1 flips | K1 is side-canonical (images are oriented to one knee side), so a flip creates an unseen input | no |
| K1 crop-shift TTA (one trained 352 px view) | -0.000085, gate +0.003 (D-B-279) | no |

### 7.6 K1 runtime: per-study fallback, fail-loud and the 8.8 h guard

- **v1-v7, fail-safe.** Any single K1 failure (load, capture, classification or guard) dropped all of K1 and silently submitted the parent CSV. The run then scored like the parent, and a score could not prove that K1 had run (k1x2sr20 probably degraded this way; D-B-249, D-B-253).
- **v8-v9, per-study fallback + fail-loud.** A failed study uses the parent's value in the K1 rank slot. A systemic failure (load failure or timeout, the guard firing, or more than 5 % failed studies) writes the parent CSV and then raises at the very end, so Kaggle shows an error instead of a silently weaker score. A scored v8+ run therefore proves K1 ran on at least 95 % of hidden studies; the first such proof was S1 (D-B-262). v9 removed a v8 defect that reverted a fully successful K1 run when the notebook finished after the guard (D-B-253).
- **v10.** A slow Kaggle session made 4 of 32 heavy studies exceed the 120 s per-study capture deadline (renders 143-151 s; the same roster passed 32/32 on a normal session), failing the 5 % limit (D-B-261). v10 raised the capture deadline to 900 s and the call limit to 300 s; the model-load limit stayed at 1500 s (registered in `experiments/final/*/registration.json`). v10 K1 outputs were bit-identical to v7 on the visible studies (D-B-277).
- **Guard.** The K1 guard stops K1 after 8.8 h from kernel start. With fail-loud a guard can only turn a late finish into an error, so it was set to the maximum rather than a conservative 8 h (D-B-253). It does not bound the parent phases or queue time, which is how U, U4 and F4 still ran past 9 h (section 10).

![K1 runtime v10: per-study fallback, fail-loud above 5 % failed studies, the 8.8 h guard, the timeouts, and the Oct 22 runs that died near 9.2 h](docs/figures/k1_runtime.svg)

*K1 runtime v10 and its limits: a scored run proves K1 ran on at least 95 % of hidden studies, but the guard only stops the K1 branch.*

### 7.7 Visible run and N32 stress run

- **Visible run.** The notebook as it will be submitted, run on Kaggle against the three public example test studies. Checked: every study ID x 12 finite outputs, every member loaded, K1 executed on every study with zero fallbacks, codecs active, fail-loud not triggered, runtime version and guard as registered, and the formula re-computed independently from the members' outputs.
- **N32.** A separate never-submitted kernel that runs the same code on 32 representative real training studies chosen to include the heaviest renders (most windows, most series). It checks every new member under load, counts per-study failures and measures per-phase time. N32 timings were dominated by model-load I/O and over-predicted hidden per-study cost (D-B-275), so hidden runtimes were calibrated from earlier hidden runs, not from N32 alone.

### 7.8 The 8-step submission gate

From Oct 21 (plan Update 3) every submission had to pass, in order: (1) independent code review by a different agent; (2) the visible run with all checks above; (3) N32 32/32 for any new member; (4) SHA256 check of every attached dataset; (5) projected hidden runtime <= 8.3 h; (6) the admission tool PASS plus a check-only replay; (7) a separate 7-item final pre-submission check by another agent; (8) an announcement in chat just before submitting. Submissions went through one tool (`kaggle_submit_once.py`) that writes a durable ledger entry before the call, makes exactly one attempt and never retries automatically.

![The 8-step submission gate as a pipeline: review, visible run, N32, dataset SHA, runtime at most 8.3 h, admission, final check, chat notice, then one submission attempt](docs/figures/submission_gate.svg)

*The 8-step gate every submission passed from Oct 21; any FAIL went back to the builder and nothing was submitted.*

### 7.9 Auto-selection as a sub-display signal

Kaggle pre-selects ("auto-selection candidates") the two submissions with the best **unrounded** public score. When several submissions display the same 0.958, which two carry the marker reveals their order below the display precision. We recorded it after every reading: for example R224 and R7 full flip > fluid routing and compo15 > C85-R7 at 0.957 (D-B-195), the K1 blends above the K1-free 0.958s (D-B-249), x3-v2 and p25 above S1 and P4 (D-B-262, D-B-282). The ordering is exact but only among the top entries, and it measures public score only.

### 7.10 Evidence ledger

`artifacts/evidence_ledger/` held two tables: `measurements.csv` (each candidate's local values on every registered source, written **before** its public reading) and `readings.csv` (public scores read later). `tools/evidence_ledger.py` reports, for each source, how many submission pairs it orders the same way as the public board (gaps < 0.002 count as ties). The retrospective table (D-B-081) gave consensus and binarised Aquila 102 of 118 agreeing pairs, relabel_v1 100, JEV 99, Dread 96, flight 93, and Gold58 unusable (2-3 readings). The ledger is why consensus and Aquila became rulers, and why no single source was trusted as a public predictor.

## 8. How the work evolved

![score timeline](docs/figures/score_timeline.png)

| Date (2026) | Milestone | Public | Private |
|---|---|---|---|
| Sep 11-12 | Copy of the public 0.941 stack (probe22 family) and five routing probes | 0.940-0.941 | 0.922-0.923 |
| Sep 13 | First own model: ConvNeXt-S, standalone | 0.909 | 0.891 |
| Oct 2 | Copy of the public 0.943 stack (jiweiliu, Mattia v34 recipe) | 0.943 | 0.927 |
| Oct 6-8 | Own models on report labels; best CoAtNet5 + RadImageNet R50 | 0.910-0.926 | 0.897-0.912 |
| Oct 9 | AUG1 augmentation for own R50 (+0.006 public, D-B-080); public OAI CoAtNet 0.949 + own | 0.931 / 0.950 | 0.916 / 0.933 |
| Oct 11 | A5q: public R + S6 + own CoAtNets + partners | 0.954 | 0.940 |
| Oct 12-13 | C85 (P956 + 15 % own), then C85-R7 with heliosli's R7 bundle | 0.955 / 0.957 | 0.939 / 0.941 |
| Oct 15-16 | R224, flip TTA, R6 swap, OA route (first private 0.943) | 0.957 | 0.941-0.943 |
| Oct 17 | H-B-189: public per-target weights on the R6 line | **0.958** | 0.943 |
| Oct 18-19 | K1 trained (v1 collapse, v2, v3-G seeds); standalone K2 | 0.944 | 0.931 |
| Oct 20 | In-pass K1 blends: x3-v2 (final A), p25 | 0.958 | 0.943 |
| Oct 21 | P25F flat-P hedge (final B); A5 DINOv3-S member | 0.957 / 0.956 | 0.943 / 0.941 |
| Oct 22 | P4 (full-bag 4th K1 member); U, U4, F4 hit the 9 h limit | 0.958 / - | 0.943 / - |

**Sep 11-13.** We started from the best public stack of the time (0.941) and tested per-finding routing probes; none moved the public score. A first own ConvNeXt-S scored 0.909. No submissions followed until Sep 30.

**Sep 30 - Oct 2.** We built the KneeOps dashboard and a data manager, re-read the public notebooks, and reproduced the then-best public 0.943 stack exactly (D-A-018). The 0.945-0.946 public notebooks depended on their authors' private assets; their public forks scored 0.943 (D-A-024).

**Oct 3-8: own models and labels.** On a fixed grouped five-fold split (4,343 studies) own models transferred to the public board at roughly their OOF size within one protocol (D-B-020, D-B-042) but stayed at 0.910-0.926 alone and added nothing to the 0.943 stack (0.942, D-B-031). Label work started here (section 3).

**Oct 8-11: Aquila and public checkpoints.** Aquila was built (Oct 8-9) and L2 labels adopted (D-B-095). AUG1 augmentation lifted own R50 from 0.925 to 0.931 (D-B-080). nartaa published an all-data CoAtNet trained with OAI images (public 0.949); blending it with our CoAtNets gave 0.950-0.951, and adding heliosli's S6 bundle gave 0.954 (A5q, D-B-137). Own OAI-masked training did not help (EXP155 0.946 = EXP143v2 0.946, D-B-101). The seven-lens readout was introduced (section 4.2).

**Oct 12-15: compact bundles and serving.** heliosli's R7 bundle raised the line to 0.957 (D-B-168). Offline GDCM decoders were added for JPEG Lossless / JPEG-LS series that the pinned reader could not decode (D-B-162, D-B-169). R224, flip TTA, fluid routing and own-share changes all displayed 0.957. Three own-recipe kill tests scored 0.926-0.948 standalone (D-B-187, D-B-190, D-B-191).

**Oct 16-17: R6 and public weights.** Swapping R7 for heliosli's R6 and applying the public per-target weights gave 0.958 (H-B-189, D-B-213). A Fracture route lowered it to 0.956; the public 2.5D reader overlay tied.

**Oct 17-19: K1.** We trained K1 (section 6.1). Its first standalone hidden runs failed with unexplained exceptions; the repaired standalone scored 0.944 (D-B-219). A sequential K1 branch after the parent would have needed about 5.2 h more and could not finish within 9 h (D-B-242), so the K3/INF3RCO overlays almost certainly published their parent.

**Oct 20-21: in-pass K1.** Sharing the R6 render made K1 affordable (D-B-243); K1 blends displayed 0.958 and were auto-selected over the K1-free parents (D-B-249, D-B-260), the first public evidence that K1 added signal. The fail-loud runtime (v8-v10) made a scored run proof that K1 executed (D-B-253, D-B-262).

**Oct 22.** Six holdout tests failed their gates and one passed; P4 scored 0.958 but ranked below p25; U, U4 and F4 timed out (D-B-282). The final pair X + F was confirmed on Oct 23 00:3x KST.

## 9. Validation

| Instrument | Size | Used for | Limitation |
|---|---|---|---|
| Gold58 | 58 studies, official labels | label-source audits, diagnostics, L7 | exposed by repeated use; weight gains on it did not transfer (D-B-246) |
| Five-fold OOF (G1) | 4,343 studies | own-model screening, Oct 3-15 | nested fitted stackers leak fold ancestry (D-B-160); OOF deltas transfer only within one protocol and label set (D-B-042) |
| Fold screens (fold 0/2/3) | 864-888 studies | seven-lens screens, Oct 11-16 | one fold per screen; seed noise of the same size as effects |
| OAI MOAKS panel | 557 knees (96 pilot + 461 confirmation) | L7 image evidence for OAI-free models | different population; only some findings map; OAI-trained arms ineligible |
| 977-study fold-0 holdout | 976 primary | K1 gates, Oct 18-22 | report-derived rulers share errors; reused many times (selection risk); models trained on fold 0 cannot be scored on it |
| Public board | about 390 studies, 0.001 display | binary decisions with large expected effects; auto-selection order | see below |
| Visible + N32 runs | 3 / 32 studies | execution, parity, runtime | say nothing about score |

**Public-board resolution (D-B-016).** Re-sampling 390-study subsets of fixed predictions gave standard deviations of the macro-AUC difference of 0.0014 for an inference-only change, 0.0028-0.0032 between two separately trained runs and 0.0037 between two different models. With a finite-population correction for drawing 390 of about 1,300 this is about +/-0.004 (inference change) and +/-0.007-0.010 (separately trained models) as 95 % ranges; picking the better of two arms by public score inflates the winner by about half a standard deviation. These are proxies from weak labels, not a confidence interval of the real public difference. The board displays three decimals. Most late changes were expected to be below 0.002, below what the public score can resolve, which is why auto-selection order and preregistered holdout gates were used, and why public numbers were never used to tune weights or thresholds.

![public vs private](docs/figures/public_vs_private.png)

**How public predicted private (our scored submissions, excluding the 37 without a description).**

| Set | n | Pearson | Spearman | Mean public - private |
|---|---|---|---|---|
| All | 87 | 0.996 | 0.972 | 0.0149 |
| Public >= 0.955 | 40 | 0.768 | 0.688 | 0.0149 |

| Public display | Private displays |
|---|---|
| 0.958 (14 submissions) | 0.943 x 13, 0.942 x 1 (R224 dropped, no K1) |
| 0.957 (18) | 0.943 x 7, 0.942 x 6, 0.941 x 5 |
| 0.956 (4) | 0.943, 0.942 x 2, 0.941 |
| 0.955 (4) | 0.940 x 3, 0.939 |

**Own models and the private board.** Our final rank was the first of the 304 teams at private 0.943, one display step above the bronze cut-off (0.942). We did not measure where the many teams that displayed public 0.958 with the public recipe landed; the evidence below is from our own submissions only. Within our own submissions, the R-family lines without the per-target weights (H-B-164/167/170/174) and the K1-free INF3 variant without R224 scored 0.942, and 9 of 11 scored K1 blends scored 0.943 (the two below contain the A5 member). These are display-rounded readings on one split, consistent with a small positive private contribution from the own members; they are not a controlled measurement.

## 10. Final week, failures and runtime

**Final selection.** X (x3-v2, public 0.958) + F (P25F, public 0.957), selected manually on Oct 23 00:3x KST under the pre-committed rule; both private 0.943. All 21 submissions at private 0.943 were tied with the final pair at display precision.

**Failures over the whole competition.**

| Date | Submission | What happened | Cause |
|---|---|---|---|
| Oct 3 | D01 (anchor + 10 % own) | hidden run error, no score | not established (D-A-035) |
| Oct 6-7 | H-B-18 control (twice) | ERROR | Kaggle system error (D-B-025) |
| Oct 11 | A6q v6 | hidden rerun failed | not established (D-B-136) |
| Oct 12 | C85 v1.0.2 | no score | not established; a zero-fallback veto and a duplicate R pass were found and removed (D-B-155) |
| Oct 18-19 | K2 (K1 standalone), two versions | hidden exceptions | not established; repaired v5 scored 0.944 (D-B-219) |
| Oct 22 | U (13:58 KST) | COMPLETE without score at <= 9.18 h | 9 h limit on a slow day |
| Oct 22 | U4 (15:59 KST) | COMPLETE without score at <= 9.18 h | 9 h limit |
| Oct 22 | F4 (22:39 KST) | COMPLETE without score at <= 9.21 h | 9 h limit |

Five further submissions without a description (Oct 9 and Oct 13) finished without a score; they are outside the logged research pipeline.

**Hidden runtimes** (submit to COMPLETE, queue included, upper bounds from 5-10 minute polling): P + O parent 6.7-6.9 h; p25 6.1-6.2 h; P25F <= 6.87 h; A5 <= 6.77 h; k1sr20 <= 7.04 h; S1 <= 7.2 h; av2 7.27-7.35 h; C <= 7.53 h; x3-v2 7.90-7.98 h; P4 <= 8.24 h; DF <= 8.29 h. DF and A5 have identical compute (only four constant weights differ) and differed by 1.52 h (D-B-280).

**Oct 22 runtime.** Three of four submissions that day ran past about 9.2 h and failed; P4, with the same compute as F4, had finished at <= 8.24 h earlier that day. Their estimated medians had been about 7.6 h (U) and 8.05 h (U4). The 8.8 h K1 guard did not prevent the failures: it is measured from kernel start and stops only the K1 branch, not the parent phases or queue time (moderate confidence, D-B-282).

**F4.** Its first build died at 18:00 KST when the Codex builder returned "model at capacity"; the failure was noticed at 19:20, the rebuild passed review at 19:52, and its 32-study qualification waited behind other kernels until 22:37 KST. It was submitted at 22:39 KST and timed out at about 9.2 h. Under the pre-committed rule F4 could only have entered the final pair together with an FB-containing A, which did not happen.

## 11. Key observations

1. 97 pipeline submissions (37 untracked uploads without a description are ignored): 87 scored, 8 finished without a score, 2 errors. 21 reached private 0.943, the best we recorded; both finals are among them.
2. Public and private moved together over the whole range (Pearson 0.996, n = 87) but much less among the 40 entries at public >= 0.955 (Spearman 0.688); public minus private averaged 0.0149.
3. Own models alone peaked at public 0.948 / private 0.935 (B172, own-only blend) and K1 alone at 0.944 / 0.931. All 21 submissions at private 0.943 contain own models (O, K1 or both) at 15-25 % of the rank weight.
4. Raising the own-model share on proxy evidence did not raise the public score in three tests (own 0.25: 0.957; per-target K1 vector: 0.957; full-bag 4th member: 0.958 below p25).
5. Of the last seven preregistered K1 holdout tests, one passed (+0.00145) and six did not (-0.0045 to +0.0022 against their thresholds).
6. Same-compute hidden runtimes differed by 1.52 h on Oct 21 and reached 9.2 h on Oct 22; three submissions were lost to the 9 h limit.
7. The seven-lens retro scan changed 16 of 74 past verdicts (D-B-129); the four-ruler and lens readouts consistently favoured own-model weight increases that the public board did not confirm (section 4.5).

Sections 12 and 13 compare this with the top teams' published write-ups and record our own retrospective (Oct 24-25).

## 12. Top-team strategies

Collected on Oct 24 from the 25 solution write-ups linked on the leaderboard and 8 discussion posts (private ranks 5 to 5,049; ranks 1-4 had not published yet). Full index, patterns and per-write-up summaries: [top-solutions-20261024.md](docs/reports/top-solutions-20261024.md) · [details](docs/reports/top-solutions-20261024-details.md).

1. **Labels from the images.** Report-only LLM labels scored about 0.89-0.91 macro AUC on gold-58, image-aware reads about 0.93-0.95 (7th, 8th, 21st, 22nd, 72nd, 77th). The image weight was set per finding: high for fracture, synovitis, Baker's, effusion and contusion, low for ligaments and the OA compartments. Every team in the top 22 that used LLM labels had a model look at the MRI.
2. **OOF self-distillation re-anchored to the fixed label.** Target = a x label + (1 - a) x OOF prediction, re-mixed with the original fused label every round (7th, 8th, 22nd, 49th, 72nd, 76th).
3. **Site habits removed in training.** A site x finding bias term in the training loss only, dropped at test: +0.003 to +0.007 private (38th, 63rd, 72nd), +0.014 for the 33rd. Folds grouped by site or scanner.
4. **A strong own model.** Whole-study reading, mm crops at 336-448 px, anatomy-anchored local views placed by segmentation or detectors, a per-finding readout that keeps medial and lateral apart. The best single models reached 0.957-0.958 private; ensembles added little at the top.
5. **Rulers that can see the change.** Label and site changes were not judged on report-derived OOF but on image-label OOF, within-site AUC, external panels, sealed folds, LB pairs with outcomes fixed in advance, and per-finding LB probes. Teams that stopped at blends of the public R6/Raptor/S6 recipe, the authors of those public models included, plateaued at private 0.942-0.944.

## 13. Self-feedback

Checked against discoveries.md and the handoff archives by four read-only investigators, then independently fact-checked: [postmortem-20261025.md](docs/reports/postmortem-20261025.md) · [evidence](docs/reports/postmortem-20261025-evidence/).

No submission of ours scored above private 0.943 (first reached on Oct 16; 21 of the 29 scored entries in the last week sat exactly there), so the final-pair choice cost nothing and the last week's variants could not move the score. Own models alone peaked at 0.935 private; the top single models reached 0.957-0.958.

| # | What we missed | Why (from our records) |
|---|---|---|
| M1 | Image-read labels never became the main label: aquila (Oct 9, all 4,405 studies) stayed a 25 % share | Higher shares were judged mainly on report-derived rulers, which cannot see the gain; D-B-166's registered rule read GO but was dropped as a holistic call; a public check of aquila was declined to save submissions; the reader kept our narrowed rules and 15 central slices |
| M2 | Rulers that could not see label, site or geometry changes (the root cause) | The limit was recorded on Oct 7 (D-B-042), yet gating stayed on report-derived rulers; the only image lens was usually ineligible; per-finding LB probes fell under our own no-diagnostics-kernel rule (Oct 13); folds were never site-grouped |
| M3 | No training-only site-bias term | One post-hoc offset test scored against Dread (D-B-002), which carries the habit itself; a site-memorisation signal from Oct 2 was recast as "no shortcut" |
| M4 | Own model stuck at 0.934-0.935 private | Kill tests changed labels, teachers and backbones but never the geometry; anatomy crops (S6+) were held on Oct 12 and valued by their diluted blend share; K1's own kill rule was not applied |
| M5 | The last week went into +0.001 work on the saturated public-recipe family | The right diagnosis was on record (reviews of Oct 13 and 16) and was withdrawn on report-ruler evidence; a "+0.001 per 1-2 days" target took over |
| M6 | Process overhead | About 55 % of tagged discoveries were infra or runtime; consume-queue mode and strict byte-diff gates blocked multi-day label -> retrain chains |
| M7 | Late start | First submission on Sep 11; the main lever was found 13 days before the end |
| M8 | Forum read for assets, not methods | Threads showing image-aware labels and 0.960 single models were closed as "no usable asset" |

Where we could have finished (estimates on the final board; gains do not add exactly):

| Scenario | Private | Rank |
|---|---|---|
| Minimal fix: image-heavy labels on our own model, blended at real weight | 0.945-0.947 | about 101-211 |
| **Realistic, same timeline from Oct 9**: plus training-only site bias, re-anchored OOF rounds, anchored crops, blend at about 0.5 | **0.948-0.951** | **about 42-100** |
| Strong execution, same timeline | 0.952-0.954 | about 22-41 |
| Full window from August | 0.954-0.956 | about 11-26 (gold) |

Lessons we take forward: build a ruler that can see the target first, and never let rulers built from the training labels judge changes to those labels; treat a large gap to the top as structural (supervision and the core model, not blend weights); score own models standalone and apply the kill rules we write; check the labels for site and language habits on day one; keep process rules proportional; read the forum for methods; start the main line in the first two weeks. The system produced the right insights several times (D-B-074, D-B-042, the Oct 13 review); our failure was in acting on them.

## 14. How we used AI agents and skills

The work was run by AI coding agents coordinated through four shared files (a "research orchestrator" layout), kept as written at the repository root ([agents.md](agents.md), [plan.md](plan.md), [discoveries.md](discoveries.md), [handoff.md](handoff.md)). Taeyang set the goals and the standing rules; the agents planned, built, reviewed, trained, ran and submitted under those rules, and every rule is dated in `agents.md`.

![Coordinator B (Claude Code) dispatches Codex astra builder and verifier lanes, consults Fable and Opus, writes the four coordination files and runs work on Kaggle, the local RTX 4070 Ti SUPER and Vast.ai](docs/figures/multi_agent_setup.svg)

*Who did what: one coordinator, separate builder and verifier lanes, independent consultants, four shared files and three compute pools.*

### 14.1 Models

| Model | Platform | Role |
|---|---|---|
| **astra 6** (`gpt-6-astra`) | Codex CLI lanes | builders astra-b1..b5, verifiers astra-1..6, overflow lanes astra-x1..x6; the consult partner astra-6; the Aquila label reader; the radiologist-persona readers |
| **Opus 5.5** | Claude Code | the coordinator session "B"; Opus subagents for the 7-item final pre-submission check, full re-reads of the findings log, and stand-in builds and reviews when Codex was unavailable |
| **sol 6.1** | Codex CLI | simple tasks only (`SIMPLE=1`): status and output collection, short smoke runs, file moves |
| **Fable 5.1** | Claude | occasional strategy consults and some reviews, alongside astra-6 |

### 14.2 Skills and tools

| Skill / tool | What it did for us |
|---|---|
| [research-orchestrator-skill](https://github.com/TaeyanG4/research-orchestrator-skill) (Taeyang's public repository) | the four coordination files (`agents.md`, `plan.md`, `discoveries.md`, `handoff.md`), the scored plan queue, one cross-check of every discovery by a different host, and the split between slots (who owns a line of work) and hosts (which platform ran it) |
| numerical-parity-skill | inference parity between Kaggle T4 and the local RTX 4070 Ti SUPER with tolerances fixed before comparing: same-checkpoint FP16 maximum probability difference 0.0004814 and AUC difference 0.000006 on 869 studies (D-A-025); at most 0.10 % of study-pair orders changed per target in a rank blend (D-A-045) |
| Kaggle skill and Kaggle CLI | kernel pushes, status and output files; every submission went through `tools/kaggle_submit_once.py` (durable ledger entry before the call, one attempt per candidate, no automatic retry) |
| Claude Code built-in browser, Claude in Chrome | reading Kaggle forum threads, the leaderboard and the GPU-quota settings page, which Codex could not render and the CLI cannot read |
| dataviz skill | the charts in this README |

### 14.3 How the agents worked together

| File | Holds |
|---|---|
| `agents.md` | operating rules and every standing instruction from the team, with dates |
| `plan.md` | one scored queue of unfinished work, Priority = 2 x Impact + 2 x Information + Confidence + Unblock + Diversity + (3 - Cost) (each 0-3), claims, and the final-slot schedule with Updates 1-12 |
| `discoveries.md` | about 390 findings (`D-<slot>-<NNN>`), each with numbers, interpretation and one cross-check by a different host (VERIFIED / CHALLENGED / HOLD / PENDING) |
| `handoff.md` | the event log: who did what, when, with which artifacts |

- **Coordinator B (Claude Code, Opus 5.5).** Planned and scheduled the queue, launched every GPU job and Kaggle push, decided and made the submissions, and was the only agent in contact with Taeyang. Subordinate lanes could not launch GPU jobs or write to Kaggle (`agents.md`, Roles/slots & Codex lanes). Slot A (Codex) wrote research proposals and consolidated reports; slots C and D ran early infrastructure and preparation work.
- **Codex lanes through a dispatch script.** `dispatch.sh` (v3.2.0 from Oct 13) sent named-file prompts to persistent lanes: builders astra-b1 (release), astra-b2 (data and training), astra-b3 (scouting), astra-b4 and astra-b5 (pool); verifiers astra-1..6; and automatic overflow lanes astra-x1..x6 when all were busy. sol took the simple tasks. Requests went through a local Docker `codex-lb` load balancer (`agents.md`, Current names; Token revocation).
- **One build per agent, independent strict review.** Each builder job built exactly one candidate from a parent already proven on the hidden set, with only the registered change plus version metadata. A different lane reviewed it before any GPU or Kaggle run, and a FAIL was binding: the builder fixed it and B never overrode it (`agents.md`, Strict builds; mistake-log item 16).
- **Consult pairs with the rule fixed first.** Major decisions (own-model share, which candidate fills a slot, the final pair) went with the same brief to two independent consultants, Fable 5.1 and astra-6, with Opus as the fallback when Fable was unavailable. Both answers, the decision and its reason were recorded, and the decision rule was written into `plan.md` before the result it depended on (`agents.md`, Important/ambiguous decisions; section 5).
- **Final check by an agent that did not build or review.** A separate Opus agent ran the 7-item pre-submission check, step 7 of the gate in section 7.8 (`agents.md`, Final pre-submission check).
- **Radiologist-persona expert panel.** On Oct 21 six single-target radiologist-persona readers (astra-1..6) went through Gold58 false-negative-like and false-positive-like cases of six weak findings (24 cases from 20 studies). They found no pipeline defect and no proven Gold58 error and proposed protocol- and coverage-related hypotheses; these led to the K1 failure map (D-B-267) and the targeted K1 tests of Oct 21-22 (section 4.4).
- **LLM relabelling.** The Aquila labels were read by astra lanes (section 3.2). The host allowed LLM labelling of the training reports provided the report text is not redistributed (D-B-029).
- **Timers instead of cron.** Session cron jobs fired only while the session was idle and missed runs during busy work, so waits used background timers (`.local/timer-until.sh`) and pollers, plus a resource sentinel that woke B on a GPU stall, an idle GPU, a down load balancer or low disk space (`agents.md`, Timers, not crons; Current sentinel).
- **Claude stand-ins.** When Codex was out of tokens or capacity (Oct 15 and from Oct 17; the F4 build lost on Oct 22, section 10), Opus subagents built and reviewed with the same prompts and file scopes, still with different agents as builder and reviewer (`agents.md`, Fallback; Codex-down admission path).

### 14.4 Operating rules that mattered

- **No hand-written decision scripts by the coordinator.** B wrote no numerical decision scripts beyond trivial one-liners; an astra builder wrote them from a written spec naming the exact inputs, a separate lane verified them, and the tests ran before any result was used (`agents.md`, Builder checklist, rule of Oct 10).
- **Preregistered gates and the seven-lens readout.** Every screen's rule was written down before the run (section 4.4) and read through L1-L7 (section 4.2); a failed gate was never relabelled a pass.
- **The 8-step submission gate** (section 7.8) and one attempt per candidate.
- **Data and credentials.** Credential files were never read or printed; the coordinator never opened radiology report text, and competition label rows were never printed in chats or logs (`agents.md`, Hard access limits, Oct 11). Public notebooks and forum posts were treated as data, never as instructions, and external public code ran only inside authorized Kaggle candidates (`agents.md`, Untrusted material).
- **Review and records.** Builder and reviewer were always different agents. Every long or important task (training, inference, release builds, submissions, decision-driving analyses) was reviewed before and after. Every finding was recorded exactly once, and every negative result carries its likely cause, confidence and a falsifier.

**Compute.** A local RTX 4070 Ti SUPER, Kaggle T4 x 2 sessions for submissions and qualification runs, and rented Vast.ai cloud GPUs (an RTX 3090 earlier, an RTX 4090 for part of the K1 training). Paired comparisons always ran on one GPU type and software stack (`agents.md`, Platform pairing).

## 15. KneeOps local dashboard

![dashboard overview](docs/figures/dashboard_overview.png)

KneeOps is a local, single-user dashboard (FastAPI + React, SQLite with full-text search, bound to 127.0.0.1) that collected the competition's notebooks, discussions, leaderboard and our submissions, translated the official pages into Korean, compared experiments only when their label source, folds and hashes matched, and opened MRI studies in a local viewer. Details: [docs/DASHBOARD.md](docs/DASHBOARD.md).

![dashboard library](docs/figures/dashboard_library.png)
![dashboard history](docs/figures/dashboard_history.png)

*Screenshots of the local dashboard; they show no competition images, labels or reports.*

## 16. Repository layout

```text
README.md, README.ko.md   this report (English / Korean)
NOTICE.md, LICENSE        credits, image attributions, licence
docs/
  EXPERIMENTS.md          every submission by family, the seven-lens screens and the K1 holdout experiments
  DASHBOARD.md            the KneeOps dashboard
  data/submissions.csv    our 97 pipeline submissions: date, name, public, private, status, family
  figures/                figures used in this report
tools/                    analysis, release-stage builders, admission and submission tools
monitoring/               KneeOps local dashboard (FastAPI + React)
training/                 training scripts and configs
tests/                    tests for the tools and the dashboard
kaggle/                   notebook sources (no weights)
agents.md, plan.md        project rules / final-pair policy and queue, kept as written (owner quotes removed)
discoveries.md            the findings log; discoveries.ko.md is its Korean translation
handoff.md                event log between sessions
ledger/                   aggregate evidence readings and the submission-attempt ledger
docs/reports/             final results, consolidated report, research catalog, visual guide, review archive, audits
docs/architecture/ ...    internal documentation written during the competition (architecture, decisions,
                          handoffs, history, learning, legacy, plans, research, runbooks)
research/                 reviews of public notebooks and ideas
experiments/final/        the two final Kaggle notebooks
```

**Not included:** competition data (DICOM images, labels, report text, study or series identifiers), per-study predictions, model weights (public or ours), contents of private Kaggle datasets, credentials, run caches and logs. Paths in the coordination files and the tools refer to the original private workspace.

## Reports and navigation

The repository also keeps the reports written at the pre-result cutoff (Oct 22, 18:15 KST) and the original research visuals. They are preserved as written; this README supersedes their "pending" states with the final results above.

| File | Purpose |
|---|---|
| [Final results](docs/reports/final-results.md) / [최종 결과](docs/reports/final-results.ko.md) | Private score and rank, the confirmed final pair, every late submission outcome |
| [Consolidated project report](docs/reports/project-summary-20261022.md) / [한국어 종합 보고서](docs/reports/project-summary-20261022.ko.md) | The full pre-result state: history, candidates, exact formulas, failures and the final-pair rule |
| [Research catalog](docs/reports/research-catalog-20261022.md) | All 390 discovery IDs with titles, dates and review fields, including negative and inconclusive findings |
| [Visual guide](docs/reports/visuals-20261022/README.md) | Four original diagrams and charts (study-to-prediction, candidate family map, public score snapshot, full-bag evidence) with source values and reproduction code |
| [Pre-submission review archive](docs/reports/pre-result-review-evidence-20261022/README.md) | The independent reviews of U, P4 and U4 (execution and formula replay) |
| [Documentation audit](docs/reports/documentation-audit-20261022.md) | Checks, corrections and remaining limits of the documentation |
| [External cleanup result](docs/reports/external-cleanup-result-20261022.md) | Storage cleanup after the competition: what was removed and what was kept |
| [Update guide](docs/reports/final-results-update-guide.md) | How `docs/reports/final-results-data.json` regenerates both results pages (`python tools/render_final_results.py`) |
| [discoveries.md](discoveries.md) / [discoveries.ko.md](discoveries.ko.md) | Canonical findings log (English) and its Korean translation |
| [handoff.md](handoff.md), [plan.md](plan.md), [agents.md](agents.md) | Event log, final-pair policy and queue, project rules and the mistake log |
| [ledger/](ledger/) | Aggregate evidence readings and the durable submission-attempt ledger |

![Conceptual MRI-study to prediction pipeline](docs/reports/visuals-20261022/study-to-prediction.png)

*Original diagram from the pre-result report (schematic icons, not patient imagery).*

## License

Our code and documents: Apache License 2.0 ([LICENSE](LICENSE)). Public Kaggle work we used keeps its authors' licences; see [NOTICE.md](NOTICE.md). The MRI example images are under their own Creative Commons licences (NOTICE.md).
