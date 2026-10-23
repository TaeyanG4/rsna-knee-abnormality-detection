# Experiment catalogue

Every submission we made to the Kaggle competition *RSNA Knee Abnormality Detection*, grouped by family, followed by the main local experiments: the seven-lens screens (Oct 11-16) and the preregistered K1 holdout tests (Oct 18-22). The overview, the label and ensemble decisions and the definitions used here (rulers, lenses, P, O, K1, w_t) are in the [README](../README.md).

- **Public** is the public leaderboard display (about 390 hidden studies), **private** the private display read after the close (about 910 studies). Both are rounded to 0.001.
- **Dates** are UTC, from Kaggle's submission list. KST = UTC + 9 h.
- The machine-readable list is [data/submissions.csv](data/submissions.csv) (date, name, public, private, status, family). Status: `scored`, `complete-no-score` (the hidden run finished but produced no score, usually an exception or the 9 h limit), `error`.
- `r(.)` is a per-target percentile rank over the full test roster with average ties.
- IDs `D-x-NNN` point to `discoveries.md`; `H-B-NNN` are plan items.

| Family | Submissions | Scored | Public | Private | Dates |
|---|---|---|---|---|---|
| A. Public 0.941 stack probes | 6 | 6 | 0.940-0.941 | 0.922-0.923 | Sep 11-12 |
| B. Public 0.943 stack + own legs | 3 | 2 | 0.942-0.943 | 0.926-0.927 | Oct 2-7 |
| C. Own models only | 31 | 27 | 0.893-0.948 | 0.881-0.935 | Sep 13 - Oct 19 |
| D. Public OAI CoAtNet + own blends | 10 | 9 | 0.950-0.954 | 0.933-0.940 | Oct 9-11 |
| E. P956 / C85 | 7 | 6 | 0.954-0.955 | 0.938-0.940 | Oct 12-13 |
| F. C85-R7 (heliosli R7) | 10 | 10 | 0.956-0.957 | 0.941-0.942 | Oct 13-15 |
| G. R6 line + public per-target weights | 9 | 9 | 0.956-0.958 | 0.941-0.943 | Oct 16-17 |
| H. INF2 / INF3 and K1-RCO overlays | 7 | 7 | 0.957-0.958 | 0.942-0.943 | Oct 18-19 |
| I. In-pass K1 blends (final families) | 14 | 11 | 0.955-0.958 | 0.940-0.943 | Oct 20-22 |
| **Total** | **97** | **87** | | | |

## A. Public 0.941 stack probes (Sep 11-12)

The best public stack of the time ("probe22" family, per-finding routing weights tuned on the public board). We copied it and tested five routing variants.

| Date | Name | Change | Public | Private |
|---|---|---|---|---|
| Sep 11 | Anchor probe22 | exact copy of the public 0.941 stack | 0.941 | 0.923 |
| Sep 12 | EXP018 | ACL/MCL inner CoAtNet weight 0.20 (ligament routing) | 0.941 | 0.923 |
| Sep 12 | EXP031 | ACL/MCL Raptor only | 0.940 | 0.923 |
| Sep 12 | EXP034 | outer-weight extrapolation | 0.941 | 0.923 |
| Sep 12 | EXP035 | outer-weight shrink | 0.940 | 0.922 |
| Sep 12 | EXP036 | + Raptor v8 reverse diversity arm | 0.941 | 0.923 |

Outcome: no routing change moved either board by more than 0.001.

## B. Public 0.943 stack + own legs (Oct 2-7)

| Date | Name | Change | Public | Private |
|---|---|---|---|---|
| Oct 2 | Anchor swap | unmodified copy of jiweiliu's fast 2 x T4 0.943 stack (Mattia v34 recipe) | 0.943 | 0.927 |
| Oct 3 | D01 | 90 % anchor + 10 % legacy own R2 ranks | - | - (hidden error, cause not established, D-A-035) |
| Oct 7 | H-B-15 | 0.8 r(anchor) + 0.2 r(own CoAtNet5 0.7 / ResNet34x5 0.3), shared decode | 0.942 | 0.926 |

Outcome: the own leg did not add to the public stack (D-B-031).

## C. Own models only (Sep 13 - Oct 19)

Models trained by us, alone or blended only with each other; no public checkpoint unless stated.

| Date | Name | Model / change | Public | Private | Outcome |
|---|---|---|---|---|---|
| Sep 13 | EXP045 | ConvNeXt-S 288 px, adjacent slices, 5 folds | 0.909 | 0.891 | first own model |
| Oct 4 | B_RN3F | own B + ResNet34, 3 folds, rank 90:10 | 0.913 | 0.895 | vs anchor 0.943: -0.030 (D-A-038) |
| Oct 4 | R2 EXP072 | 224 px, mixed report/teacher labels | 0.902 | 0.890 | teacher leakage (D-A-021) |
| Oct 4 | R1 EXP061 | 224 px, report labels (control for R2) | 0.904 | 0.889 | R2 not better |
| Oct 6 | H-B-13 B224 | single split, CoAtNet 224 | 0.910 | 0.897 | transfer test |
| Oct 6 | H-B-13 R384 | single split, Raptor 384 | 0.913 | 0.898 | +0.003 vs B224, as in development (D-B-020) |
| Oct 6 | H-B-12 E1 | equal rank mean of B224 and R384 | 0.917 | 0.904 | +0.004 over the better member (D-B-021) |
| Oct 6 | H-B-18 student | 0.5 Dread + 0.5 ResNet34 inner-OOF labels, BF16 | 0.893 | 0.881 | |
| Oct 6-7 | H-B-18 control (x2) | matched BF16 control | - | - | ERROR twice, Kaggle system error (D-B-025) |
| Oct 7 | H-B-19 A | CoAtNet-2 224, 5 folds, Dread | 0.914 | 0.897 | |
| Oct 7 | H-B-19 B | CoAtNet5 0.7 + ResNet34x5 0.3 | 0.914 | 0.899 | +0.0030 OOF, 0 public (D-A-049) |
| Oct 7 | H-B-20 | distilled labels (0.5 Dread + 0.5 OOF mean) | 0.912 | 0.898 | no gain (D-B-027) |
| Oct 7 | H-B-28 | definition-aligned relabel_v1 labels | 0.913 | 0.900 | inconclusive (D-B-038) |
| Oct 8 | H-B-31 B (x3) | RadImageNet ResNet-50, 5 folds, alone | 0.925 | 0.912 | +0.011 vs CoAtNet5 (D-B-049); three identical submissions |
| Oct 8 | H-B-31 A (x2) | CoAtNet5 + RadR50x5 equal rank | 0.926 | 0.912 | most of the +0.0076 OOF complement did not transfer (D-B-057) |
| Oct 9 | H-B-34 AUG1 | AUG1 augmentation RadR50, 2 seeds x 5 folds, plane-aware flip TTA | 0.931 | 0.916 | +0.006 public, about half the fold-0 gain (D-B-080) |
| Oct 9 | EXP143v2 own | own CoAtNet 384 (0.949 public recipe without OAI), L2 labels, SWA 14/11/12 | 0.946 | 0.934 | |
| Oct 10 | EXP155 own | EXP143v2 recipe + OAI images on masked PF OA / Lateral OA / Synovitis | 0.946 | 0.933 | OAI gave no visible gain (D-B-101) |
| Oct 10 | B172 | 0.70 r(EXP143v2) + 0.30 r(L2 partner bundle) | 0.948 | 0.935 | best own-only submission |
| Oct 11 | B4 | 0.35 own S42+S43 + 0.35 EXP174 + 0.30 partners | 0.946 | 0.933 | |
| Oct 14 | H-B-151 | EXP143v2 recipe on soft targets 0.7 R7 + 0.3 own OOF | 0.944 | 0.927 | DEAD vs 0.946 (D-B-187) |
| Oct 15 | H-B-156 | ConvNeXt-S on the EXP143v2 contract, L2 labels | 0.926 | 0.911 | DEAD (D-B-190) |
| Oct 15 | H-B-157 | public R fine-tuned 6 epochs on 0.7 R7 + 0.3 own-OOF soft targets | 0.948 | 0.932 | DEAD vs parent 0.949 (D-B-191) |
| Oct 16 | H-B-177 | ConvNeXt-S recipe fix | 0.934 | 0.924 | +0.008 vs H-B-156, still DEAD (D-B-202) |
| Oct 18-19 | K2 (x2) | K1 v2 standalone | - | - | hidden exceptions, cause not established |
| Oct 19 | K2 v5 | repaired K1 v2 standalone | 0.944 | 0.931 | (D-B-219) |

## D. Public OAI CoAtNet + own blends (Oct 9-11)

"Public OAI 0.949" is nartaa's all-data CoAtNet SWA checkpoint (later R384). S6 is heliosli's compact ConvNeXt-S bundle.

| Date | Name | Formula | Public | Private |
|---|---|---|---|---|
| Oct 9 | B3-AUG1 | 0.8 r(public OAI 0.949) + 0.1 ConvNeXt reader + 0.1 AUG1 RadR50 | 0.950 | 0.933 |
| Oct 9 | B3-H1 | same with the H1 partner (AUG1 RadR50, 5 heads) | 0.950 | 0.933 |
| Oct 9 | exp152 blend | 0.5 r(EXP143v2) + 0.5 r(public OAI 0.949) | 0.951 | 0.936 |
| Oct 10 | blend3 | 1/3 each: public OAI 0.949, EXP143v2, EXP155 | 0.951 | 0.937 |
| Oct 10 | EXP169 v1.0.2 | 0.85 r(blend3) + 0.15 r(L2 partner bundle) | 0.951 | 0.936 |
| Oct 10 | A172 | 0.85 r(blend3) + 0.15 r(L2 RadR50 + ConvNeXt-T + DINO-base) | 0.951 | 0.937 |
| Oct 11 | A4 | 0.85 r(public 0.949 + own S42/S43 + EXP155) + 0.15 r(partners) | 0.951 | 0.937 |
| Oct 11 | A5 (v5q) | 0.85 r(P954 + own + EXP155) + 0.15 r(Q), P954 = 0.70 R + 0.30 S6 | 0.952 | 0.938 |
| Oct 11 | A5q | 0.85 r(R + S6 + EXP143v2 + EXP155, equal) + 0.15 r(Q) | 0.954 | 0.940 |
| Oct 11 | A6q v6 | DINO removed, own S42/S43 added | - | - (hidden rerun failed, D-B-136) |

Outcome: adding S6 lifted the line from 0.951 to 0.954 (D-B-137).

## E. P956 / C85 (Oct 12-13)

| Date | Name | Formula | Public | Private |
|---|---|---|---|---|
| Oct 12 | P956 | 0.5 r(R) + 0.5 r(S6) | 0.954 | 0.938 |
| Oct 12 | C85 v1.0.2 | 0.85 r(P956) + 0.15 r(own O) | - | - (no score; D-B-155) |
| Oct 12 | P956-R+ | R continued on L2 labels, 50/50 with S6 | 0.954 | 0.939 |
| Oct 12 | C85 v1.0.4 | 0.85 r(P956) + 0.15 r(O = EXP143v2 + EXP155 + Q) | 0.955 | 0.939 |
| Oct 12 | P956-R+2 | two R+ seeds, 50/50 with S6 | 0.954 | 0.939 |
| Oct 13 | vNext-A | 0.70 r(P956-R+2) + 0.30 r(O_A) | 0.955 | 0.940 |
| Oct 13 | vNext-B | 0.70 r(P956-R+2) + 0.30 r(O_B) | 0.955 | 0.940 |

Outcome: vNext-A/B tied C85; multiple simultaneous changes prevent attributing anything to the own weights (D-B-167).

## F. C85-R7 (Oct 13-15)

R7 is heliosli's compact ConvNeXt-S bundle `8c7b666e`. P = 0.5 r(R) + 0.5 r(R7) unless stated.

| Date | Name | Change | Public | Private |
|---|---|---|---|---|
| Oct 13 | C85-R7 v1.0.0 | 0.85 r(P) + 0.15 r(O) | 0.957 | 0.941 |
| Oct 13 | C85-R7 + codecs v1.1.0 | offline GDCM decoders for JPEG Lossless / JPEG-LS in every branch | 0.957 | 0.941 |
| Oct 13 | vNext-R7-A | 0.70 r(P) + 0.30 r(O_A) | 0.956 | 0.942 |
| Oct 14 | compo15 | upgraded own bundle O_A at 0.15 | 0.957 | 0.942 |
| Oct 14 | fluidR | Effusion and Baker's from R only | 0.957 | 0.941 |
| Oct 14 | own25 | 0.75 r(P) + 0.25 r(O) | 0.957 | 0.942 |
| Oct 14 | p65 | P = 0.35 r(R) + 0.65 r(R7) | 0.956 | 0.942 |
| Oct 15 | FLIP_ALL | R7 full flip TTA (hidden <= 6.3 h, D-B-194) | 0.957 | 0.941 |
| Oct 15 | H-B-164 | R slot = 0.65 r(R384) + 0.35 r(R224) | 0.957 | 0.942 |
| Oct 15 | H-B-167 | H-B-164 + R7 full flip | 0.957 | 0.942 |

Outcome: auto-selection ranked R224 and FLIP_ALL above fluidR and compo15, and those above C85-R7 (D-B-195).

## G. R6 line + public per-target weights (Oct 16-17)

| Date | Name | Change | Public | Private |
|---|---|---|---|---|
| Oct 16 | H-B-174 | H-B-167 with R7 swapped for heliosli's R6 `fe055406` + full flip | 0.957 | 0.942 |
| Oct 16 | H-B-170 | H-B-167 + anatomical mirror TTA on R224 | 0.957 | 0.942 |
| Oct 16 | H-B-168 | R224 + FLIP_ALL + fluid routing + compo15 | 0.957 | 0.941 |
| Oct 16 | H-B-181 | H-B-174 + hard OA route (Lateral OA / PF OA from the compact slot) | 0.957 | **0.943** |
| Oct 17 | **H-B-189** | H-B-174 + public per-target weights (Lateral OA 0.2, PF OA 0.6, Synovitis 0.8, Baker's 0.35) | **0.958** | 0.943 |
| Oct 17 | H-B-185 | H-B-181 + fluid route | 0.957 | 0.943 |
| Oct 17 | H-B-193 | H-B-181 + Fracture route from R | 0.956 | 0.943 |
| Oct 17 | H-B-195 | H-B-189 + R224 mirror | 0.957 | 0.943 |
| Oct 17 | H-B-190 | H-B-189 + goodpjw2008 2.5D ConvNeXt reader overlay | 0.958 | 0.943 |

Outcome: first 0.958 (H-B-189, auto-selected first; D-B-213); first private 0.943 was H-B-181.

## H. INF2 / INF3 and K1-RCO overlays (Oct 18-19)

| Date | Name | Change | Public | Private |
|---|---|---|---|---|
| Oct 18 | C4 | H-B-189 with five fixed mixed-scale ConvNeXt-T members in O | 0.958 | 0.943 |
| Oct 18 | INF3 | H-B-189 with per-checkpoint inverse class-cost pooling of Q | 0.958 | 0.943 |
| Oct 18 | K3-RCO10 | 90/10 overlay of H-B-189 and K1 v2 | 0.958 | 0.943 |
| Oct 18 | INF2 | Small ConvNeXt replaces DINO on eight targets | 0.958 | 0.943 |
| Oct 19 | INF3RCO | INF3 + K1 v2, 90/10 | 0.958 | 0.943 |
| Oct 19 | N224 | INF3 with R224 weight 0 | 0.958 | 0.942 |
| Oct 19 | N224O25 | N224 with own share 0.25 | 0.957 | 0.943 |

Outcome: the K1 overlays could not finish K1 within 9 h after the parent and most likely published the parent (D-B-242). N224 ranked below INF3 by auto-selection (D-B-246).

## I. In-pass K1 blends (Oct 20-22)

| Date | Name | Formula | Hidden h | Public | Private |
|---|---|---|---|---|---|
| Oct 20 | k1sr20 | r(0.8 r(INF3) + 0.2 r(K1 v2)), shared R6 render, FP16 | <= 7.03 | 0.958 | 0.943 |
| Oct 20 | k1x2sr20 | K1 = v2 + v3-G seed 1 | <= 7.04 | 0.958 | 0.943 |
| Oct 20 | **x3-v2 (final A)** | K1 = v2 + v3-G seed 0 + seed 1, stride 3, runtime v5 | 7.90-7.98 | 0.958 | 0.943 |
| Oct 20 | p25 | r(0.75 r(P) + 0.25 r(K1 trio, stride 2)), no O | 6.12-6.20 | 0.958 | 0.943 |
| Oct 20 | av2 | INF3 + K1 trio stride 2 with a per-target K1 share vector (0.1-0.5) | 7.27-7.35 | 0.957 | 0.943 |
| Oct 21 | S1 | x3-v2 + R6 eval stride 2 + runtime v9 fail-loud | <= 7.2 | 0.958 | 0.943 |
| Oct 21 | C | p25 recipe with K1 members v2 + fold models f1/f2/f3 + all-data | <= 7.53 | 0.957 | 0.943 |
| Oct 21 | **P25F (final B)** | p25 with flat P weights, runtime v10 | <= 6.87 | 0.957 | 0.943 |
| Oct 21 | A5 | P' = r(0.85 r(P) + 0.15 r(A5 5-fold DINOv3-S)) | <= 6.77 | 0.956 | 0.941 |
| Oct 21 | DF | A5 with flat P | <= 8.29 | 0.955 | 0.940 |
| Oct 22 | U | r(0.5 r(x3-v2 form) + 0.5 r(p25 form)), K1 stride 2 | failed <= 9.18 | - | - |
| Oct 22 | P4 | p25 + full-bag fold-0 K1 as a 4th member | <= 8.24 | 0.958 | 0.943 |
| Oct 22 | U4 | U with the 4-member K1 | failed <= 9.18 | - | - |
| Oct 22 | F4 | P4 with flat P | failed <= 9.21 | - | - |

Outcome: K1 blends were auto-selected over the K1-free 0.958s (D-B-249); dropping O did not hurt (D-B-260); raising K1 above 0.25 per target, fold-model K1 members, A5 and flat P each lost display steps on public; flat P lost none privately.

## Not included

The team account also holds 37 uploads without a description (Sep 30 - Oct 14, file name `submission.csv`), temporary or teammate submissions outside our research pipeline. They are left out of this catalogue and of `data/submissions.csv`.

## Seven-lens screens (Oct 11-16)

Lens definitions are in the README ([section 4.2](../README.md#42-the-seven-lenses)). Rulers: mean(JEV, Consensus) for L1/L2 unless stated; brackets are paired bootstrap 95 % intervals. "Reg. gate" is the experiment's own preregistered rule, kept beside the holistic label.

| ID | Screen | Key lens readings | Reg. gate | Label |
|---|---|---|---|---|
| D-B-129 | Retro scan of 74 past contrasts (27 screens) | 16 verdicts flip; ConvNeXt-S weak four +0.0066, macro +0.0013 | - | ConvNeXt-S re-test (H-B-105) |
| D-B-131 | Five-fold bundle without DINO | weak four +0.0020, macro +0.0006, positive CIs | - | keep R50 + ConvNeXt-T + MIL |
| D-B-132 | ConvNeXt-S MIL, fold 0, two seeds | K36-flip weak four +0.0014, macro +0.0010, harm PASS | - | moderate ADVANCE |
| D-B-134 | MIL-T coverage bags | weak four +0.00001, macro -0.0006 | - | HOLD / DROP |
| D-B-156 | Coverage sampler, own CoAtNet | L1 +0.0008, L2 -0.0003, L4 ACL +0.0115, L5 disagree, L6 -0.0003, L7 OAI +0.0103 / +0.0073 | HOLD | PER-TARGET (ACL) |
| D-B-158 | L4 implementation audit | old tool passed PF OA with 2 of 4 rulers >= +0.003 | - | L4 corrected (v2 tool) |
| D-B-161 | Cross-slice feature interactions | weak four -0.0006, macro -0.0011, harm guard fails | - | DROP |
| D-B-164 | Laterality-canonical MIL-S | weak four -0.0013, macro -0.0009; OAI only +0.0081 | - | DROP |
| D-B-165 | Pairwise weak-four ranking loss | weak four -0.0022, macro -0.0020, MCL -0.0118 | - | DROP |
| D-B-166 | Aquila 60 % on five targets | weak four -0.0041, macro -0.0021; Gold58 weak five -0.005; OAI +0.006 | - | DROP |
| D-B-180 | Strong-teacher soft pseudo-labels | macro +0.0030 but weak four -0.0002; image lenses slightly negative | NO_GO | NO_GO |
| D-B-184 | Native 320 px input, fold 1 | macro +0.0029 [+0.0007, +0.0054], all rulers positive; one seed/target cell below -0.005 | NO_GO | NO_GO (floor) |
| D-B-189 | Serving-time scale TTA | mean - 320 = -0.00004, 0/4 rulers CI > 0 | NO_GO | NO_GO |
| D-B-192 | Knee-centred compartment crops | macro -0.0076 / -0.0005 / -0.0048 / -0.0035; Lateral OA +0.0108, PF OA -0.0174; L7 Gold58 +0.0041 | NO_GO | NO_GO |
| D-B-196 | SAM continuation of own CoAtNet | four-ruler mean -0.0027 vs matched AdamW; both seeds negative; all eligible lenses fail | DEAD | DEAD |
| D-B-197 | OA-only crop routing, fold 3 | L1 +0.0018, L4 Lateral OA +0.0079 (4/4), L5 +0.0001, L6 +0.0003; Medial OA -0.0071 | FAIL | DEPRIORITISED |
| D-B-198 | Multi-view inputs (full field + compartment crops), fold 2 | L4 PASS (+0.0074), L6 +0.0015 [+0.0005, +0.0023]; fold 3: macro -0.0017, L6 +0.0006 | NO_PASS | ADVANCE, then HOLD |

## Local holdout experiments (K1, 977-study fold 0)

K1 trained on folds 1-3 (3,190 studies); fold 0 (977 studies, 976 in the primary cohort) held out. Deltas are K1-only macro AUC, candidate minus control, on JEV / Aquila / Dread / Consensus. Every rule was written to a file before the run, and every result was re-computed by a different agent.

| ID | Date (KST) | Change | Rule | Result | Status |
|---|---|---|---|---|---|
| D-B-218 | Oct 18 | K1 v2 vs heliosli R7 fold 0 | within -0.005 on >= 3/4; Spearman <= 0.93 | -0.0010 / -0.0001 / +0.0131 / +0.0028; Spearman 0.777 | KEEP |
| D-B-244 | Oct 20 | OAI native-grade adaptation vs equal-update RSNA control | - | mean -0.00004, Spearman 0.999 | OAI not useful |
| D-B-245 | Oct 20 | average of v2 and v3-G seed 1 vs v2 | all rulers > 0 | +0.0015..+0.0029, P(>0) >= 0.97 | adopted (2-checkpoint K1) |
| D-B-247 | Oct 20 | eval stride 2 vs 3 | - | trio +0.0007..+0.0017 on 4/4, P(>0) >= 0.95 | adopted in the p25 family |
| D-B-252 | Oct 21 | post-processing from past RSNA winners (sex prior, cross-target pairs, rank-Gauss) | all 4 rulers > 0 and Gold58 P(>0) >= 0.8 | 0 / 13 pass | NO_GO |
| D-B-264 | Oct 21 | full-bag study-level loss (all windows, gradient replay) replacing the 64-window cap | >= 3/4 and mean >= +0.002 | -0.0080 / -0.0037 / -0.0036 / -0.0026 (mean -0.0045), although training loss was lower (0.364 vs 0.394) | NO_GO |
| D-B-268 | Oct 21 | eval stride 1 vs 2 | >= 3/4, no target < -0.005 | -0.00011 / +0.00013 / +0.00004 / +0.00003 | GO, effect about 0, not submitted |
| D-B-269 | Oct 21 | target-type pooling at inference (focal targets: + top-3 windows; extent targets: + window mean) | >= 3/4, mean >= +0.002, no target < -0.005 | mean -0.0030 | NO_GO |
| D-B-271 | Oct 22 | Lateral OA crop expert, 0.5 / 0.5 rank with the trio on Lateral OA only | mean Lateral OA >= +0.006 | +0.0031 / +0.0019 / +0.0054 / -0.0015 (mean +0.0022) | NO_GO, positive direction |
| D-B-272 | Oct 22 | failure-weighted retrain (agreed FN/FP x 2, disputed x 0.5) | >= 3/4 and mean >= +0.002 | -0.0024 / +0.0011 / -0.0036 / -0.0023 (mean -0.0018) | NO_GO |
| D-B-273 | Oct 22 | Synovitis head refit with equal loss per effusion-prior bin | Synovitis >= 0 on 3/4 and mean >= +0.006 | Synovitis mean -0.0107 (macro -0.0009) | NO_GO |
| D-B-278 | Oct 22 | full-bag checkpoint as an additional 4th member (stride 3) | all >= 0 and mean >= +0.001 | +0.00085 / +0.00159 / +0.00201 / +0.00136 (mean +0.00145) | GO -> P4: public 0.958 below p25, private 0.943 |
| D-B-279 | Oct 22 | crop-shift TTA (one trained 352 px view at y=8, x=16) | mean >= +0.003 | -0.00017 / -0.00002 / -0.00011 / -0.00004 | NO_GO |

The same checkpoint was -0.0045 as a replacement (D-B-264) and +0.00145 as an additional member (D-B-278).

## Other local results referred to in the README

| ID | Result |
|---|---|
| D-B-016 | public-board resolution: SD of differences 0.0014 (inference change) to 0.0037 (different models) on 390-study subsets |
| D-B-017 | one same-machine seed change: macro -0.0060 (MCL -0.066) |
| D-B-042 | within one protocol and label set the Dread OOF delta transferred to public at slope about 1 (public - OOF = 0.0166 +/- 0.0005); across families it failed |
| D-B-066 | AUG1 RadR50 fold-0 two seeds: Dread +0.0141 [+0.0096, +0.0198], all rulers and targets positive |
| D-B-081 | retrospective ruler agreement with public order: Consensus and Aquila 102/118 pairs, relabel_v1 100, JEV 99, Dread 96, flight 93 |
| D-B-095 | L2 labels: five-fold two-seed RadR50 JEV +0.0050 [+0.0038, +0.0062] |
| D-B-139 | corrected-OAI CoAtNet fold 0: macro -0.0078, weak four -0.0099, all ruler CIs negative |
| D-B-181 | R7 full flip TTA on 989 exposed held-out studies: +0.0013 [+0.0003, +0.0023] |
| D-B-250 | R6 stride 2: inference -35 % on 32 studies, per-target Spearman 0.983-0.999 |
| D-B-261 | v9 capture deadline 120 s: 4/32 heavy studies failed on a session about 2x slower |
| D-B-267 | K1 OOF failure map: 87 % agreed cells; FN 260 / FP 1,227; co-finding pockets |
| D-B-280 | same-compute hidden runs: A5 <= 6.77 h vs DF <= 8.29 h |
