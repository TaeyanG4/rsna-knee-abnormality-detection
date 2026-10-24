<!-- Investigator report for postmortem-20261025.md (B, 2026-10-25). Read-only review of discoveries.md, handoff archives, plans and memory; citations are file:line or D-IDs as found. -->

> **Correction (2026-10-25, B):** the brief given to this investigator said "35 submissions from 10-16 to 10-22 all scored private 0.943". The correct count is 34 submissions, 29 scored: 21 at 0.943 (the maximum) and the rest at 0.940–0.942 or own-only tests. Read "the 35 submissions" below with that in mind.

## 1. Our rulers and which gates used them

| Ruler | How it was built | Report-derived? |
|---|---|---|
| Dread | Public LLM soft labels from reports; the training labels of most early models | Yes |
| JEV | Public OpenJev text reads plus a Steven/Pilkwang swap; its authors disclose Gold58 tuning (agents.md:187) | Yes |
| flight_v4 | Hybrid built from Steven and Pilkwang report labels | Yes |
| relabel_v1 | Our own Opus re-read of the report text (D-C-001) | Yes |
| consensus | Dread scored only on cells where Dread, Steven v4, Pilkwang v2 and relabel_v1 agree. Agreement with Gold58 is 0.709 on Synovitis, because "all reports share the same silence" (D-B-035) | Yes |
| aquila | A vision LLM given the images plus flight_v4 labels plus report text (D-B-074/078/081). L2 trains on it, so it is self-referential for L2 models (D-B-097) | Partly: image-aware but anchored on report labels |
| Gold58 | The 58 official image reads; exposed | No |
| OAI MOAKS panel | 557 external OAI knees read on images, mapped to about 9 targets. Counted only for models with no OAI training in their history | No |
| 976/977 holdout | R7's hash fold 0, scored on JEV, aquila, Dread and consensus (D-B-248). Exposed, and in-sample for the public R branch (D-B-210) | Yes |

How the gates evolved:
- **10-07/08:** Dread OOF gates (Tier A: Dread gain ≥ .004 with CI > 0 and consensus ≥ 0; agents.md:203).
- **10-09:** JEV primary with a consensus guard (EXP149, D-B-081).
- **From 10-11:** weak-four and macro readouts on mean(JEV, consensus) with a Dread harm guard. Lenses L1–L6 are all report rulers (agents.md:205–207). L7, the only image lens, was usually "not eligible", e.g. D-B-166, and for the final K1 work "L7 independently eligible image evidence is missing".
- **Final week:** the four 976 report rulers plus Gold58 P>0 in pre-fixed rules (D-B-252, D-B-278), then the public LB.

## 2. Folds, and whether site leakage was measured

- **ADR 0004 (10-02):** report-group 5-fold split (docs/decisions/0004-validation-protocol.md:16).
- **D-A-003:** report groups are not patient-independent; identical pixel data appeared across folds. This led to the G1 contract of 4,196 components.
- **G1 split:** train 3,455 / validation 888, assigned by hashing the component ID (tools/exp084_group_loss.py:66–77).
- **EXP099:** a greedy whole-component partition, balanced on Dread positive counts. `scanner_group` is only carried as a column (tools/exp099_kfold.py:12–14, 58).
- **Final week:** R7's study-hash folds, which split 42 duplicate groups (D-B-176). Holdout = 977 studies with those excluded, giving the 976 cohort.
- **Never site- or scanner-grouped.**
- ADR 0004:24–25 required results sliced by scanner group and a pre-specified scanner-group holdout for the final recipes. Only per-group slices were ever run (R2: 0.856–0.927, docs/handoffs/2026-10-03-0600-r2-result.md:16). The audit states plainly that no scanner holdout was done (docs/research/oof-error-contribution-2026-10-03.md:55). It never happened later. The only leave-one-protocol-out figure is buried inside an unrelated test in D-B-215 (−0.016 / −0.018).

## 3. Site, scanner and language reporting habits; what D-B-002 tested

- **10-02 forum triage:** a metadata-only model scored 0.652 with random folds vs 0.598 with scanner-grouped folds, which our note read as "0.05 = site memorisation". 13 vendor × field-strength groups; "is the public/private split by site?" unanswered. Planned action: watch scanner-group OOF (docs/research/discussion-triage-2026-10-02.md:20–21).
- **D-B-013 (10-07)** recast the same fact as "(no shortcut)". The report language is mentioned only as "a multilingual Report column".
- There is **no positive-rate-by-site or by-language analysis** anywhere in our records.
- **D-B-002, exactly:** a per-group, per-finding additive logit offset applied to stored B and R384 predictions. Cross-fitted on 5 component folds inside V888. Groups: field, scanner_group, manufacturer, cached_mask. Two estimators: label-free centering and a label-aware class-mean offset. Scored against rounded Dread labels.
  - Every variant was negative (−0.0005 to −0.0054), except cached_mask label-aware (+0.0013 / +0.0015).
  - First conclusion: "metadata calibration closed as a cheap lever". Codex narrowed it to "these two estimators on reused weak V888 labels; other calibration forms untested".
- **A training-only site-bias term was never proposed.** The nearest idea was astra's 10-13 #5, protocol-group risk, which grouped by acquisition completeness and was "not a scanner offset" (.local/idea-round-20261013/astra/PROPOSALS.md:111–124). It became H-B-139, deferred at priority 9, and was never run (plan-archive-20261023.md:88).
- **D-B-252** *added* a PatientSex prior (the opposite of the 228th's subtraction); it was negative.

## 4. Did any ruler track the public LB?

- **D-B-042 (10-07):** 13 rulers vs 12 submissions, Spearman 0.74–0.83, none could be preferred. Dread tracked public at slope ≈1 only within one protocol and label set. A label change (relabel5) landed +0.017–0.030 above every own-label reference. The written implication was "label changes must not be judged by any own-label reference".
- **D-B-081 (10-09), retrospective:** consensus and aquila_bin both agreed on 102/118 pairs, relabel 100, JEV 99, Dread 96, flight 93. Adding images moved flight from 93 to 102, the one hint that image-checked labels track the official reads.
- **D-B-118 (10-11):** the per-target map was inconclusive (36/47 vs 38/47 pairs, p = 0.5).
- **What came of the ledger:** the last readings.csv entries are on 10-14 (00:46 handoff event). Nothing was recorded for the 35 submissions of 10-16 to 10-22. By then the scores moved by less than the 3-decimal display (0.956–0.958).
- **D-B-259 (10-21):** on the two known LB-ordered pairs, every report lens and all four rulers got the weight-change pair wrong. The lenses correlated with our rulers at r .97–.99.
- **Verdict:** no ruler that tracks the LB was ever found. The ledger's pairs were mostly same-label variants of our own models, where report rulers do work, and almost never label or site changes.

## 5. Per-finding LB probes and LB pairs with pre-fixed outcomes

- **Probes: never run.** Fable's 10-13 proposal P1 was exactly the top-team method: a weak-four "classwise" probe in 2 slots, reading the weak-four mean as W = 3s − 1, with a decision rule fixed in advance (.local/idea-round-20261013/fable/PROPOSALS.md:26–34).
  - It became H-B-120 and was CLOSED on 10-13 at 18:04 as "public probes conflict with the Kaggle qualification-only rule" (handoff-archive-20261013-14.md:1157).
  - That rule was Taeyang's 10-13 16:2x GPU-quota rule; its "probes" were decoder and parity kernels.
  - The 10-14 02:59 amendment allowed constant-only variants without their own visible run, which a probe is. H-B-120 was never revived.
- **LB pairs with pre-fixed outcomes: partial.**
  - Done: pre-registered forecasts (D-B-042); a written gain review before each submission; D-B-016's policy of a pre-registered public guard for label changes.
  - Never done: an LB pair for any label or site change. Pure aquila (L1) and the 60 % aquila share were killed on report rulers before ever reaching the LB.

## 6. Concrete cases

**Rejected by report rulers, valued by top teams:**
- **D-B-124:** report-silent soft priors. Synovitis fell on every report ruler "as their silence = negative encoding predicts" → HOLD. The 74th and 38th gained from treating "not mentioned" differently from negative. Report rulers cannot reward this by construction.
- **D-B-166:** a 60 % aquila share on Effusion, Synovitis and the three OA compartments was dropped. Report rulers negative, OAI panel +0.006, Gold58 also negative on Synovitis. Astra's correction says the registered rule read GO; the DROP was B's holistic call.
- **D-B-083 L1** (aquila only): HOLD on JEV −0.0058 / Dread −0.0178, but Gold58 was also negative (−0.0098), and Astra under-calls MCL and PF OA. This rejection may have been partly right.
- **D-B-002:** site offsets rejected on Dread labels.

**Admitted on report rulers, no visible LB or private effect:**
- **D-B-095** (L2 ADOPT, JEV +0.0050): public B3-H1 0.950 = B3-AUG1 0.950.
- **D-B-248** share vector: best on all four rulers and Gold58 P>0 .84, yet LB 0.957 < 0.958 (D-B-251).
- **D-B-246:** weight changes favoured by Gold58 lost on the LB.
- **D-B-245/247/278** (K1 trio, stride 2, full-bag 4th member P4): gains of +0.0007 to +0.0020 on the 976 rulers, yet every one of the 35 submissions scored private 0.943. Caveat: differences under 0.0005 are hidden by the 3-decimal display.

## Misses

**M1. Training-only site-bias term**
- **Top teams:** a bias term in the loss only, dropped at test: +0.005 to +0.014 private.
- **Us:** one post-hoc offset test (D-B-002). H-B-139 was not about site and was deferred.
- **Why:** not considered, plus wrong ruler. D-B-002 scored against Dread, which carries the habit itself, so removing a site offset must lower AUC against it. Differences between groups were assumed to be "real prevalence" (D-B-002 text). D-B-013 read "no shortcut".
- **Confidence:** high.
- **Falsifier:** a per-site positive-rate table showing no spread beyond what Gold58 would predict, or a training-only bias term on K1 giving no LB or within-site gain.

**M2. Site-grouped folds and a leakage check**
- **Top teams:** scanner/site-grouped folds; found random-fold OOF inflated by 0.019–0.07.
- **Us:** component or hash folds only. The scanner holdout planned in ADR 0004 was never run.
- **Why:** process rule not carried out. After 10-04 the focus moved to label sources and blends.
- **Confidence:** high that it was never done; medium that it mattered.
- **Falsifier:** a scanner-grouped re-split showing the same ranking of candidates.

**M3. Report-derived rulers deciding label changes**
- **Top teams:** judged label changes on image-label OOF, external panels and LB probes.
- **Us:** report-derived gates (L1–L6, the 976 rulers), although we had written down that they cannot judge label changes (D-B-042, D-B-112, D-B-014, agents.md:204). L7 was mostly "ineligible" because of the history rule.
- **Why:** wrong ruler and process. The ancestry rules disqualified the only image lens, and we had no locked image panel.
- **Confidence:** high.
- **Falsifier:** an LB probe of the D-B-166 or D-B-124 arms that also came out negative.

**M4. Per-finding LB probes**
- **Top teams:** kept or inverted one column to read each finding's public AUC (49th, 8th, 102nd).
- **Us:** proposed (fable P1), then closed (H-B-120).
- **Why:** a process rule misapplied (the quota rule), plus D-B-016's overfitting caution extended to diagnostic probes.
- **Confidence:** high.
- **Falsifier:** a rules or quota record showing probes were in fact barred. A paired standard error of about 0.01 would also have limited what a weak-four probe could resolve.

**M5. The evidence ledger**
- **Top teams:** measured noise floors and fixed acceptance rules in advance.
- **Us:** the ledger stopped after 10-14 and never covered label or site changes.
- **Why:** resolution. The 3-decimal display and ±0.004–0.010 public noise (D-B-016) swamped the differences, and attention moved to serving work.
- **Confidence:** medium.
- **Falsifier:** ledger records for 10-15 to 10-22 that do discriminate between candidates.
