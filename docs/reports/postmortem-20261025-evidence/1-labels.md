<!-- Investigator report for postmortem-20261025.md (B, 2026-10-25). Read-only review of discoveries.md, handoff archives, plans and memory; citations are file:line or D-IDs as found. -->

## 1. What aquila was

- **Reader:** Astra (Codex gpt-6-astra), using "variant F" (EXP148, `tools/exp148_aquila_full.py`).
- **Inputs, one exam per call:**
  - 15 central slices shown as montages (the EXP146 renderer);
  - the report text, piped in by a script;
  - the flight_v4 report label (probability plus extraction confidence);
  - the relabel v1.1 rules pasted in word for word (D-B-078, discoveries.md:2379; handoff-archive-to-20261012.md:772).
- **Report-aware and anchored to the report label.** Astra's review confirmed the ancestry runs flight_v4 → aquila (discoveries.md:2760).
- **Built:** 2026-10-09 06:35–11:41 KST. All 4,405 exams were read. After removing Gold58, aquila_v1.csv has 4,347 rows (handoff-archive-to-20261012.md:772, :784, :1051).
- **Quality on Gold58 (macro AUC):** full-run reading 0.909 against 0.899 for flight_v4 alone, so about +0.01. The pilot reading of the same exams gave 0.9187. Top teams moved from about 0.89–0.91 to 0.93–0.95.
- **cygnus_v1 / lyra_v1 / orion_v1 (full Opus / Sol reads and their combination) were never produced.** Only aquila files exist in storage/labels/derived_constellation/.
  - Opus over 4,405 exams (about $2–4k) was judged "not justified" (D-B-078 :2381). Codex reviewer A challenged that: Opus F had the best Brier and log loss (D-A-051 :2559).
  - The user decided "full Astra F run first, then Sol E if tokens remain" (:772). Sol E was never run.

## 2. Was an image-anchored label set ever trained at full weight and scored on the LB?

**No.**
- **L1 (aquila only)** was trained on RadR50, folds 0+1, locally only (EXP149, D-B-083 :2753). It lost: JEV −0.0058, Gold58 −0.0098, MCL −0.054.
- **No public check:** B proposed a paired public check (EXP150 on aquila). The user declined: "<owner remark removed>" (to save submissions, since local verification was possible) (2026-10-09 08:43, :893).
- **Highest share that reached the LB:** 25 % (L2), as EXP143 v2: public 0.946, private 0.934 (D-B-190 :4111). A fifth aquila head (H1) appeared only at 0.1 blend weight, B3-H1 = 0.950, which did not move the LB (D-B-083 update).
- **Local-only higher shares:** 50 % was flat (D-B-123). A per-target 60 % share was dropped (D-B-166).

## 3. Why higher image shares were rejected, and whether anyone questioned the rulers

**L1 (EXP149).** The pre-registered rule used JEV as primary, consensus as guard and Gold58 ≥ −0.005. JEV is a report-derived proxy (astra-6, :2758). Consensus shares sources with Dread. Flight is an input to aquila.

**D-B-166 (2026-10-13).** The proposal (scout E5) said in advance: "Our report-derived rulers (JEV/Dread/consensus) cannot see this gain and penalise it by construction" (.local/plan-collection-20261013/external/PROPOSALS.md:59). Its go rule was Gold58 + OAI image panel, with report rulers only as a harm guard (:65).
- **Result:** the OAI image-read panel was positive (+0.006, seed 42); Gold58 was −0.005 (seed 42, inside the ±0.012–0.016 noise).
- **Decision:** B dropped it on the report rulers (weak-four −0.0041, macro −0.0021).
- **Correction:** astra-1 later found the registered rule read **GO_RULE_MET**. "DROP is B's holistic decision, not a registered-rule failure" (:3824, 2026-10-14 01:16). The item was never reopened.

**The rulers were questioned repeatedly at the time:**
- astra (2026-10-09 08:28): the risk is "teaching the wrong clinical definition while correlated references reward it" (:857).
- D-B-097: aquila is the one reference L2 trains on.
- D-B-112 (10-10): for Synovitis and Contusion no report-derived ruler is adequate.
- D-B-124 (10-11): Synovitis falls on report rulers "as their silence = negative encoding predicts".
- D-B-259 (10-21): lenses built from report labels over-credit our own models, which are trained on report labels.

**Retro re-measure (D-B-118)** still used report-derived "map rulers".

## 4. Blockers (with handoff timestamps)

- **Perceived cost rule (2026-10-07):** H-B-37 was registered at Priority 7, "low-priority", with the note that the "reasonable cost" rule (D-B-029) "rules out whole-corpus image reading" (plan-pre-unify-20261009.md:183). Top teams read all studies (72nd: about 45 minutes).
- **OpenAI approval:** "Sending all training images to OpenAI (Sol) needs the user's go" (D-B-074, 10-09 ~02:04). It was granted in effect, since Astra is also OpenAI.
- **Codex tokens:**
  - The first aquila run hit a usage limit at 07:34 on 10-09 (:784).
  - Codex credits were exhausted on 10-13 at 06:08 (handoff-archive-20261013-14.md:449).
  - Outages from 10-15 12:47 and 10-17 00:28 onward; no new admissions without Codex receipts.
  - A second full read (Sol E ≈93 M tokens) would have competed with the build and review lanes for the same pool.
- **User priorities:** "<owner remark removed>" (finishing the Astra run as the first goal) (10-09 08:18, :797). Consume-queue mode from 10-10 19:57: no new exploration items, "Codex tokens ran out once today" (:2947).
- **KneeCoT ban:** a host ban on an external dataset (D-B-063). It had nothing to do with our labelling.
- **Duplicate-work rule:** on 10-19, D-B-235 ruled that "Repeating Opus/Astra F without a specific changed mechanism would overlap settled work" and "AI-only re-reading is not a substitute" for human readers (:4764–4766).

## 5. When label work stopped, and what replaced it

- **Formal deferral, 2026-10-12:**
  - 00:43: H-B-100 (label correction) deferred, "no valid Gold58 tuning route before freeze" (:4805).
  - 03:17: capacity consult says "STOP silence priors". C85 registered as 0.85 × public P956 + 0.15 × own models (:5060).
- **Last label tests:** D-B-166 (10-13 15:28), D-B-180 (10-14), D-B-187 (10-15, public 0.944, recorded as DEAD).
- **Replaced by:** public-checkpoint blends (R7, S6, P956), then our K1 members served inside the public blend at 0.20–0.25, plus runtime and serving engineering (D-B-242–281). The 10-22 full re-read (D-B-277) did not reopen the label line.

## 6. Top-team techniques against our record

- **"Not mentioned" ≠ negative:**
  - D-C-006 found that Dread has no mention-state field; it ended with "insufficient evidence to launch a mask fit".
  - Silent-cell fills (D-B-119/124/130) were held or dropped, judged on rulers that encode silence as negative.
- **Re-levelling to Gold58 rates:** never done. The rule "must not optimize to exposed Gold58" (D-C-002) and astra's advice to calibrate only on a non-Gold image panel (:857) blocked it.
- **Per-finding fusion:** one try (D-B-166), on the wrong targets for the image read (see the table).
- **Report-blind reader:** none. All readers saw the report label, and from variant F also the text. The user's memory rule "labelling-inputs-give-everything" (10-09) made this the default.
- **OOF distillation re-anchored to a fixed label:**
  - Single round on Dread, 0.5/0.5 (D-B-027, public 0.912).
  - R7 teacher 50/50 with L2 (D-B-180, NO_GO).
  - 0.7 × R7 + 0.3 × own OOF with **no label anchor** (D-B-187).
  - The proposed "orion_v1" (fusion with our OOF, :2381) was never built.

## Misses

| # | Top teams | Us | Why we missed it | Conf. | Falsifier |
|---|---|---|---|---|---|
| M1 | Image read as the main label; LB-checked | aquila only ever at 25 %; L1 never submitted (§2) | **Wrong ruler**, plus the user's choice to save submissions (10-09 08:43) | High | A record of an L1/high-share model scored on the LB, or local evidence that it lost on image-read rulers |
| M2 | Host rubric; 7th team used a verify-the-report prompt | Our narrowed relabel v1.1 rules (D-C-002, 10-07). Diagnosed as the cause of misses before the full run: MCL 7/9 positives missed, PF OA 8/21. The decision was "keep the validated prompt" (:772). D-A-051 (04:01) had recommended a "corrected, versioned official-definition" contract. | **Process rule** (no untested prompt change) plus **judgement** | High | A Gold58 re-read with corrected rules showing no MCL/PF OA gain |
| M3 | Whole study or full mosaic, crop/zoom | 15 central slices by a registered "cheapest within 0.005" rule. 96 slices was +0.0026 (D-B-077). The user asked "<owner remark removed>" (wasn't it supposed to be the whole study?) (:363) | **Resource / process rule** | Med | A whole-study re-read not beating 15 slices on Gold58 and OAI |
| M4 | Per-finding weights: image high for fracture, synovitis, Baker's, effusion, contusion; report kept for OA and ligaments | 60 % image share on Effusion, Synovitis, PF OA, Lateral OA, Medial OA, i.e. three OA compartments where reports win. It also imported aquila's under-calling (mean shifts PF OA −0.042, Synovitis −0.073). D-B-123's 50 % arm showed gains for Contusion, Fracture and Lateral Meniscus, but this was dismissed as fold noise. | **Judgement** (forum-driven target choice), **wrong ruler** | Med | The same test on Fracture/Contusion/Baker's/Effusion losing on the OAI and Gold58 panels |
| M5 | Report-blind reader adds errors that don't track the report's | None; the input rule required the report (memory, 10-09) | **Process rule / not considered** | High | Any blind-reader arm in the records |
| M6 | "Not mentioned" kept separate; levels mapped to Gold58 rates | Silent fills judged on report rulers; Gold58 tuning banned | **Process rule**, **wrong ruler** | Med | A fold-safe Gold58 re-levelling tried and found neutral |
| M7 | Distillation anchored to the fixed fused label, several rounds | One round on Dread; R7 teacher without the image label | **Not considered** (no fused label existed) | Med | — |
| M8 | Label work carried through to the end | Stopped 10-12/13 under the pre-freeze schedule, Codex outages, consume-queue mode, and the duplicate rule (D-B-235) | **Resource + process rule** | High | Records showing Codex capacity unused 10-13 to 10-22 with a label item still ranked |
