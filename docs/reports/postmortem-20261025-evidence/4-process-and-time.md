<!-- Investigator report for postmortem-20261025.md (B, 2026-10-25). Read-only review of discoveries.md, handoff archives, plans and memory; citations are file:line or D-IDs as found. -->

## Post-mortem: process and time allocation, RSNA Knee

**Bottom line**
- **The label lever was found early and then misjudged.** On 10-09 we showed that LLM reads with the images beat report labels on Gold58 (D-B-074, `discoveries.md:2319`). The same day we built a full 4,405-study image-aware set (aquila_v1). It stayed a 25 % share. Every test of a bigger share was scored on report-derived rulers (plus the exposed Gold58), which cannot see such a gain. It was declined on 10-13, 10-16 01:59 and 10-16 02:12. The main cause was the measurement rule and our judgement, not compute.
- **Training-time site bias was never considered.** The only site test was a post-hoc per-scanner offset on 10-06 (D-B-002, `discoveries.md:959`). It ended "metadata-conditioned calibration is closed as a cheap lever" (later narrowed).
- **After 10-13 the process steered effort into deploying a blend built on the public recipe.** About 55 % of discoveries and about 84 % of handoff events are process, infra or runtime work.

### 1. Timeline
- **08-05 → 09-10: no own record exists.** The August-dated files are timestamps of third-party label datasets. D-A-044 dates EXP001–059 only as "09-11~13; no host record" (`discoveries.md:927`). Why the first submission came on 09-11 is not recorded. Confidence low. Falsifier: Kaggle submission history or Taeyang's memory.
- **09-11 → 09-16:** weight and routing variants of the public 0.941 family plateaued at 0.940–0.941. The 09-13 reassessment (`research/reports/20260913_plateau_reassessment.md`) said "+0.015 needed" and planned clean own 5-folds. The last files are EXP055/057 on 09-16.
- **09-17 → 09-30:** no files at all, so no record. Cause unknown.
- **Phases after that:**

| Dates | What happened | Private |
|---|---|---|
| 10-01–02 | Built the KneeOps localhost web app (about 25 handoff docs), 265 GB download; anchor 0.943 resubmitted | |
| 10-03–05 | Leakage audits; biohub review preserved (10-03 13:46); agent rules v2 | |
| 10-06–09 | Orchestrator migration; parity and seed-noise work; own models 0.910–0.926 public; vision-LLM label pilots → aquila_v1 → L2 labels; EXP143v2 | 0.936 |
| 10-10–12 | Partner families; OAI; expert and astra loops; consume-queue mode; S6 releases | 0.940 |
| 10-13–15 | C85-R7 0.957 public; pseudo-label pivot NO_GO, then DEAD on public; strict gates | 0.942 |
| 10-16–22 | Single-change stacking; public 0.958 weights (H-B-189); K1 own model; K1 serving/runtime v7–v10 | 0.943, unchanged |

### 2. Overhead
**Machinery built:**
- agents.md is 86 KB, plus 275 KB of superseded rules and map archived on 10-15.
- `tools/` holds 963 .py files and there are 551 tests. That includes 25 strict_variant_gate versions, 9 kaggle_submit_once versions, 11 watchdogs and the admit tools.
- dispatch.sh reached v3.2.0 (with dispatch_runtime.py).
- Lanes: Codex astra-b1–5, astra-1–6, overflow x1–6, plus Claude opus lanes.
- 20 cross-check directories, 6 direction-review directories, 11 scout rounds, 3 expert rounds, the evidence ledger, the L1–L7 readout and the 8-step gate.
- `.local/astra-improve/builds` holds 1,744 files. Its 256 dispatch records cover only 10-12 to 10-14: about 148 have gate/review/fix/runtime-type names, 9 are pseudo-label jobs, none are site.

**Shares** (the discovery index tags live in `discoveries.md:47ff`; the research catalog has no tags):
- **Discoveries (392 tagged rows):** 214 (55 %) carry infra/release/runtime/tooling/hidden/parity tags. 50 (13 %) carry label tags. There are no site tags.
- **Label-tagged discoveries by period:** 23 before 10-10, 11 in 10-10→13, 3 in 10-14→16, 3 in 10-17→22. None of the last three trains a new label set.
- **Handoff:** 1,132 events from 10-08 23:15 to 10-24. A keyword classification (heading + Action line, roughly ±10 points) gives:

| Category | Share |
|---|---|
| Process / coordination | ≈46 % |
| Infra / runtime / release | ≈38 % |
| Model science | ≈12 % |
| Label | ≈3 % |
| Site | <1 % |

- **Our own reviewers saw this at the time.** Fable on 10-13: since 10-10, most GPU hours and ~10 Codex lanes went to partner arms worth ≤4 % of the CSV (`.local/strategy-1013/fable/REPORT.md:17`). Taeyang on 10-11 16:20: "5 hours of only verification fixes" (`handoff-archive-to-20261012.md:4064`).

### 3. Process decisions

**(a) Gold58 never used for training or gating; report-derived rulers as gates** (`agents.md:187`)
- What happened: H-B-36, the report-state → gold-rate re-levelling that 38th/5th used, was dropped on 10-09 14:05 as a "weak detour" (`handoff-archive-to-20261012.md:1260`).
- The aquila share was tested only on report rulers:
  - 0 %: −0.0069
  - 50 %: flat (D-B-123)
  - 60 % per target: dropped, although the OAI image panel rose +0.006 (D-B-166, `discoveries.md:3814`)
  - 100 %: −0.0058 JEV (D-B-083)
- No aquila-heavy model was ever scored on public.
- Effect: largest. Why: process rule plus judgement. Confidence high. Falsifier: any LB record of an aquila-majority or gold-relevelled model.

**(b) Consume-queue mode, 10-10 19:57** (`…:2944`)
- What happened: it stopped the 6 h rounds, the expert and improvement loops, and new items. This came 5½ h after L2 was adopted (10-09 14:34) and while the full-run plan was fresh (Sol B + Astra F, D-B-078).
- Broad research reopened only on 10-12 11:48 (`agents.md:214`). The 10-12 re-plan then scored items against our own 10-18 member freeze and deferred "OAI teacher → label correction → student" (D-B-133).
- Effect: delay, not a block. Why: token shortage (Codex exhausted on 10-10). Confidence medium.

**(c) Kaggle kernels only for submissions, 10-13 16:29**
- What happened: Kaggle T4 stopped being a training lane (EXP162/184 had trained there). The quota went to visible and N32 release runs. Together with the D-A-010/D-B-016 ban on public-LB probing, per-finding LB probes (used by 49th and 8th) were never available.
- Effect: it shrank own-model capacity to the 4070 plus Vast credit (about $4–9 on 10-13). Confidence medium-low.

**(d) Mandatory review (10-13), strict builds with binding FAILs (10-14 22:50), plan = candidate (10-15), 8-step gate (10-21)**
- What happened: every item had to be a byte-diff of the parent that had already completed on hidden, and a submittable kernel with a Ready-by date.
- Effect: favoured patches of the blend dominated by the public branch (P); a multi-day label → retrain → serve chain could not qualify. Why: reaction to real failures (an import-lock deadlock burned ≈3.5 GPU h). Confidence medium.

**(e) Codex outages** (10-15 12:06–15:17; 10-16 02:28; from 10-17 00:28)
- What happened: cross-checks paused; the strict gate needed Codex receipts until the Claude recorder took over at 10-17 01:36.
- Effect: hours of deployment delay; no label or site item was waiting. Confidence medium.

**(f) 6 h rounds**
- What happened: the 10-16 scout prompt asked for concrete changes in high-vote notebooks and framed the direction check as "daily single changes + one long-shot still right?" (`.local/idea-scout/20261016-0519/scout-prompt.md`). The 10-16 11:19/17:19 parts were missed, 10-17 was light or schedule-only, and nothing ran after 10-17 17:19.
- Confidence medium.

**(g) Expert panel, 10-10**
- What happened: E2 advised "Prioritize reliable inputs … over more label variants" (`.local/astra-improve/expert-r1b-1642/E2.md:1`). The loop's expert audits found real input-routing issues, and no round raised site bias.

### 4. Biohub lessons
- The lessons were already preserved on 10-03, with a weekly check asking whether tooling had pushed out research (`docs/runbooks/strategy-review.md`). I found no weekly-check event on 10-10 or 10-17.
- They were re-shared 10-16 01:57, together with the owner target "+0.001 per 1–2 days" (`handoff-archive-20261015-17.md:1231`).
- Fifteen minutes later the debate concluded "daily single-change tests + ConvNeXt-S recipe fix". F-v2 relabel was declined partly because "rulers cannot see it" (`:1267–1272`). That is the reverse of "fix validation".
- **Acted on:** stacking small gains (10-16 22:5x audit, H-B-188 bundle).
- **Not acted on:**
  - Structural change.
  - A final pair from different families: X and F both have P at 0.75–0.85, and F is the flat-P twin.
  - Mining top-ranked forum comments.
- **The next 7 days:**
  - H-B-189 transferred the public-tuned 0.958 weights. The biohub caution about public-tuned weights is recorded next to it (`plan-snapshot-20261018-1609.md:332`).
  - K1 v1 collapsed (≈12.5 GPU h lost). K1 standalone (the K2 kernel) scored 0.944, below its own ≥0.950 keep rule.
  - D-B-242→281 were serving, runtime, stride and full-bag work. Of the 74 discoveries from 10-17 to 10-22, 36 carry infra tags and 3 carry label tags.
  - D-B-256 and D-B-277 concluded "no ≥10 % lever".
- Confidence high.

### 5. Forum monitoring
- **733965** (LLM ruling), read 10-07 (D-B-029): LLM use allowed.
- **733826** (annotation guidance), read 10-07: we built the definition-aligned relabel_v1, which scored 0.913 vs 0.914 (D-B-038) and was dropped.
- **746792** ("Reaching 0.949 with one CoAtNet"), read 10-09 (D-B-079): we copied the OAI masking and window count; nothing further on labels.
- **747658** (LLM label benchmark): listed under "no host ruling" on 10-10 (`handoff-archive-to-20261012.md:2263`). Read 10-16, verdict "No new production label asset authenticated" (`.local/idea-scout/20261016-0519/REPORT.md:106`).
- **748111** ("Single sub 0.960"): "no usable authenticated member or recipe" (`:111`).
- **Site threads:** none among the 319 scouted topics.
- **749936** ("public parts only → 0.958") became our final weight recipe.
- Effect: method signals were filtered out by an "is there a downloadable asset or member?" lens. Why: scout design plus token limits (the 12 h watch was replaced by the 6 h rounds on 10-10 01:16). Confidence medium-high.

### 6. Did any reviewer flag the gap?
- **10-13, fable** (`.local/strategy-1013/fable/REPORT.md:9,15,17`): top teams "own stronger single models trained on better labels"; "Our label work stopped at L2"; the verification apparatus is "excellent at not fooling us and poor at moving the score". Astra-3 the same day: engineering "improved correctness and weak-label fit more than independent image-truth performance".
  - Outcome: we ran a pseudo-label pivot anchored to the L2 report labels. It was NO_GO on report rulers (D-B-180, 10-14 19:18) and DEAD on public, 0.944 vs 0.946 (D-B-187). Astra-3 then advised "stop fill/PL" (10-14 19:30).
- **10-15 00:04, fable:** "stop ±0.001 tweaks". The pseudo-label student it backed also died.
- **10-16 01:43, fable:** L3 labels plus an own-heavy pivot, "the only lever with ≥ +0.003 upside".
  - Outcome: withdrawn at 01:59 and 02:12 because aquila_v1 already existed and its dose-response was ≤0 on report rulers.
- **10-19, D-B-235 (expert MRI relabel):** parked; there are no MSK readers and "no AI-reader substitute" (`plan-archive-20261023.md:59`).
- **10-20:** fable's VLM student was parked (<30 % success estimate, `:70`). The Synovitis report-state ruler was parked under the no-report-text rule (`:69`).
- **Site bias:** never flagged anywhere I could find. Cause: not considered after D-B-002. Confidence high. Falsifier: any plan or review text proposing a training-time site term.

**Caveats:** `.local/scratch` was deleted on 10-23, so the 10-21 research-lane outputs could not be checked. The handoff shares come from keyword counts. No files were edited.
