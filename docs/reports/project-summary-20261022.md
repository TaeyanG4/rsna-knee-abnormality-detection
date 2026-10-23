# RSNA Knee Abnormality Detection — consolidated pre-result record

**English main** · [한국어 해설](project-summary-20261022.ko.md) · [Visual guide and sources](visuals-20261022/README.md)

**[Results and final selection](final-results.md)** are maintained separately.
This page preserves the research state at the cutoff below; it is not live run status.

**Cutoff: 2026-10-22 18:15 KST. Version 1.0.0. Results expected 2026-10-23 KST, as stated by Taeyang.**

This report consolidates existing research, engineering, submissions and decisions.
It creates no experiment, efficacy claim, independent review or final selection.
The recorded public scores below were supplied by Taeyang; no leaderboard or score
query was made for this report. Private scores, private rank and the final selected
pair remain unconfirmed. Submission acceptance and hidden completion are separate.

The full record contains **390 unique discoveries**: 52 from A, 281 from B, 54
from C and 3 from D. The [complete catalog](research-catalog-20261022.md) indexes
every discovery. [discoveries.md](../../discoveries.md) remains the authoritative
home for findings and review history; [handoff.md](../../handoff.md) records events;
[plan.md](../../plan.md) retains the selection policy and outstanding claims.

## 1. Position at the cutoff

![Conceptual flow from an MRI study to 12 target predictions](visuals-20261022/study-to-prediction.png)

*Original schematic, not patient imagery. It illustrates the input/model/output
concept; each candidate executes only its registered branches.*

| Item | Recorded state |
|---|---|
| Task | Knee MRI study classification; 12-target macro AUC |
| Best supplied public display | **0.958**, shared by several candidates; not a unique winner or a private-score forecast |
| Latest reported automatic pair | X (`57415972`) and p25 (`57427331`), both 0.958; reported screenshot, not a live selection check |
| Precommitted fallback final pair | X + F (`57475409`, 0.957), for a hedge against the four public-tuned P weights |
| Accepted last-day submissions | U `57510427`, P4 `57512878`, U4 `57514281`; no outcome/score available in the records at this cutoff |
| Next candidate owned by B | F4, the flat-P counterpart of P4; preparation/review planned or underway, no accepted submission ref recorded |
| Final decision time | Latest plan: only COMPLETE, scored candidates eligible at **2026-10-23 08:00 KST**; manual selection still needs confirmation |
| Competition close in project rules | **2026-10-23 08:59 KST**; distinct from the user's expected result date |
| Resources | No resource action in this documentation task; older GPU/credit/quota observations are historical, not live availability |

The old October 19 transfer digest and device schedule describe an earlier state.
They must not cause H216 or H210 to restart: the OAI comparison and recovered seed1
completed later (D-B-244/245). B's active ownership is preserved. This report does
not stop ongoing F4 work or restore the paused heartbeat.

## 2. Candidate identities and outcomes

![Recorded public displays and separately marked unrecorded outcomes](visuals-20261022/public-score-snapshot.png)

*Taeyang-supplied rounded public displays at the report cutoff. Unrecorded outcomes
are not zero scores. The dot-plot axis is 0.950–0.960; no private result is inferred.*

All kernels in this table belong to `taeyangg4`; all listed accepted remote versions
are **version 1**. The `-v2` suffix in X's slug is part of the slug, not remote version 2.
Refs and submission times come from the immutable attempt ledger; its PENDING
status is the status at acceptance, not a live outcome. Later user-supplied outcomes
belong to the cited discoveries.

| Name | Kernel slug | Submission ref | Submitted, KST | Public result / state at cutoff | Evidence |
|---|---|---|---|---|---|
| X | `knee-lb-r6w958k1x3sr20-v2` | 57415972 | 10-20 10:04:32 | 0.958; fallback finalist A | D-B-249/260/276; plan Updates 5-10 |
| p25 | `knee-lb-r6w958k1n3s2p25-v1` | 57427331 | 10-20 16:12:20 | 0.958; latest reported automatic selection with X | D-B-260/276 |
| F / P25F | `knee-lb-r6w958k1n3s2p25flatpv10-v1` | 57475409 | 10-21 18:23:00 | 0.957; precommitted hedge B | D-B-274 |
| D / A5 | `knee-lb-r6w958k1n3s2p25a5v10-v1` | 57484279 | 10-21 22:40:30 | 0.956; not a finalist under the registered rule | D-B-275/276 |
| DF | `knee-lb-r6w958k1n3s2p25a5flatpv10b-v1` | 57500332 | 10-22 08:40:49 | 0.955; excluded from final pair | D-B-280/281 |
| U | `knee-lb-r6w958k1uxpv10-v1` | 57510427 | 10-22 13:58:57 | Accepted; hidden outcome and score not recorded | handoff 13:59; ledger |
| P4 | `knee-lb-r6w958k1n4fbp25v10-v1` | 57512878 | 10-22 15:20:09 | Accepted; hidden outcome and score not recorded | D-B-278; handoff 15:20; ledger |
| U4 | `knee-lb-r6w958k1u4fbv10-v1` | 57514281 | 10-22 15:59:21 | Accepted; hidden outcome and score not recorded | handoff 15:59; ledger |
| F4 | Exact final slug/version/ref not yet recorded here | — | — | Preparation only at cutoff; intended P4 + flat P weights | plan Update 10; handoff 17:57 |

Earlier important readings remain in the source record: H189 0.958 (D-B-213),
reader-overlay H190 0.958 and mirror H195 0.957 (D-B-217), repaired K2 v5 0.944
(D-B-219), per-target K1 vector av2 0.957 (D-B-251), S1 0.958 (D-B-262),
fold/all-data ensemble C 0.957 (D-B-263). The shared display does not establish
equal exact scores, private equivalence, or execution of every hidden branch.

## 3. What the final candidates compute

![Registered components and candidate-family relationships](visuals-20261022/candidate-family-map.png)

*Percentages summarize nested rank fusion; the exact formulas and stride distinction follow.*

`r` denotes the registered per-target rank operation with average ties over the
full inference roster. These formulas explain the registered structure; exact
precision, ranks, model hashes and rendering behavior remain in the versioned code
and reviews. Symbols X and Y below are formula components of U and should not be
confused with the separately submitted X candidate's stride-3 implementation.

- **P:** the public branch, including R384/R224/R6flip. The registered public
  target ratios are preserved unless a candidate explicitly uses flat P.
- **O:** the prior own-model bundle (EXP143v2, EXP155 and Q); INF3 combines P and O.
- **K3:** rank aggregation of K1 v2 + K1 v3-G seed0 + seed1. p25/U use stride 2;
  the submitted X reference uses stride 3.
- **K4:** K3 plus the full-bag fold-0 checkpoint as a fourth member, not a
  replacement for seed0. D-B-278 tested this distinction.
- **Flat P:** the four tuned target ratios become 0.5 (Lateral OA 0.2,
  PF OA 0.6, Synovitis 0.8 and Baker's cyst 0.35 in the public-weight parent).

| Candidate family | Registered structure |
|---|---|
| INF3 | `r(0.85 r(P) + 0.15 r(O))` |
| Submitted X | `r(0.80 r(INF3) + 0.20 r(K3 at stride 3))` |
| p25 / F | `r(0.75 r(P) + 0.25 r(K3 at stride 2))`; F uses flat P and runtime v10 |
| U | `X_u = r(0.80 r(INF3) + 0.20 r(K3))`; `Y_u = r(0.75 r(P) + 0.25 r(K3))`; `U = r(0.50 r(X_u) + 0.50 r(Y_u))`, K3 stride 2 |
| P4 / F4 | p25 / F formula with K4 instead of K3; F4 is conditional on actual qualification/submission |
| U4 | U formula with K4 instead of K3 |
| A5 / DF | Replace P by `r(0.85 r(P) + 0.15 r(A5))` inside p25; DF also uses flat P |

## 4. Research progression and durable findings

These are summaries of existing source findings, including negative and
inconclusive work. They do not upgrade PENDING, HOLD or CHALLENGED cross-checks.

| Phase / question | What was learned | Source |
|---|---|---|
| Public anchor and early own models | Anchor reproduction reached 0.943. Weak local gains did not reliably survive public evaluation; an early own B/RN blend scored 0.913 versus anchor 0.943. | D-A-018/030/038/044 |
| Input and serving efficiency | Shared decode/render reduced duplicated work while preserving inputs; runtime must be measured jointly, not obtained by summing isolated accuracy gains. | D-B-018/048/110/243 |
| Supervision | 75% Dread / 25% Aquila L2 supervision improved five-fold JEV across R50, ConvNeXt-T and DINO-base. The labels remain correlated report-derived rulers. | D-B-095/098/111 |
| Augmentation and external/public branches | AUG1 R50 public 0.931 vs 0.925; A5q 0.954 vs A5 0.952. C85-R7 reached 0.957; later weight transfer reached H189 0.958. | D-B-080/137/179/213 |
| More components are not automatically better | vNext tied or lost to its parent; mirror and later A5 additions lost on public. Small proxy gains cannot be added to predict a joint final gain. | D-B-167/179/217/221/236/276/281 |
| OAI | Earlier OAI paths were neutral/adverse. The actual native-grade H216 OAI/control pair completed: mean OAI-minus-control approximately -0.00004 across four rulers; registered OAI_NOT_USEFUL. Neither arm replaced the original K1. | D-B-101/139/240/244 |
| K1 recovery and diversity | v1 collapsed; v2 trained successfully. v3-G seed1 recovered under its original contract. Checkpoint rank averaging improved all four 976-study rulers; the trio then added small gains over the pair. | D-B-214/218/220/245 |
| K1 serving became feasible | Sequential K1 risked exceeding the hidden budget. Sharing the R6 render passed 32-study tensor/FP32 parity and FP16 checks, enabling executed K1 blends. | D-B-242/243/249 |
| Denser inference | K1 stride 2 improved the four rulers relative to stride 3; stride 1 later changed little. R6 stride 2 reduced measured inference cost but did not establish a better public result. | D-B-247/250/262/268 |
| Public-weight caution | Increasing O or applying a per-target K1 share vector looked attractive locally but lowered the public display. Removing O with the p25 recipe retained 0.958. | D-B-246/251/259/260 |
| Fold/all-data expansion | The C five-member fold/all-data blend scored 0.957 versus p25 0.958. The existing holdout cannot independently evaluate models trained on those same rows. | D-B-263; H224/H231 |
| Full-bag objective | As a seed0 replacement it lost on all four rulers, mean about -0.0045. As an additional fourth member it gained +0.00085 to +0.00201, mean +0.001449, with independent result replay. This justified P4, not a claim of private gain. | D-A-052; D-B-264/278 |
| External scouting and label fusion | The final source survey found no demonstrated high-probability 0.959 lever. The 11-label fusion source remained highly correlated with existing rulers and did not fix their public ordering error. | D-B-256/259/265/277 |

![Full-bag replacement and addition have different local AUC effects](visuals-20261022/fullbag-local-evidence.png)

*These are two different comparisons: full-bag versus capped seed0, and K4 versus
K3. The addition test used the same 977-study fold-0 holdout at stride 3. Both bars
show K1-only local effects, not measured final-blend/private gains. Sources and
exact aggregate values: [visual guide](visuals-20261022/README.md#4-full-bag-comparison).*

The last bounded K1 tests were also retained: target-type pooling mean macro
-0.0030 (D-B-269); Lateral OA expert +0.00222 in that target, below its +0.006
gate (D-B-271); failure-weighted retraining mean -0.00179 (D-B-272);
effusion-stratified Synovitis refit mean Synovitis -0.01069 (D-B-273); and a trained
crop-shift TTA mean macro -0.000085 (D-B-279). The small positive crop-expert signal
is a future research reserve, not a broadly disproven direction. None of these
failed gates was silently relabelled PASS to fill submission slots.

## 5. Qualification, failures and remaining uncertainty

The three final-day accepted candidates have preserved independent pre-submission
reviews: [U](pre-result-review-evidence-20261022/U-final-check.md),
[P4](pre-result-review-evidence-20261022/P4-final-check.md) and
[U4](pre-result-review-evidence-20261022/U4-final-check.md).
They record actual visible and 32-study execution, all expected K1 members,
finite output, embedded-module/config checks and independently recomputed formulas.
These historical PASS decisions do not certify unseen hidden output or private gain.

Key operational lessons:

1. K2's two original hidden exceptions remain unproved root causes. The repair
   received the fuller failure-directed 4,407-header / 32-real-study validation,
   and repaired v5 later scored 0.944. Preserve failed exact versions (D-B-219).
2. Model loading sometimes exceeded the old 120-second call timeout, disabling
   K1. Later runtimes separated load/classification limits. v9/v10 added
   per-study fallback and fail-loud behavior; a successful configured run supports
   the registered at-least-95% K1 execution condition (D-B-245/253/261).
3. The suspected v9 capture-timeout mechanism was challenged: discarded underlying
   exceptions and fallback-inclusive timing prevent definitive causal attribution.
   v10's larger limits are a mitigation, not proof of the original cause (D-B-261).
4. DF's first build used a local schema that differed from the notebook's embedded
   schema and failed on `p_weight_mode`. Fresh-process imports of the actual embedded
   payload became a required stage check (handoff 10-22 03:11 and later final reviews).
5. 32-study cache parity did not prove whole-cohort identity: the later full comparison
   found 3 of 4,167 studies with different pixels; the remainder of the metadata
   differences were explicitly classified META_EPS (D-B-254/270).
6. Same-compute hidden elapsed upper bounds differed by **1.52 hours**: A5 <=6.77 h,
   DF <=8.29 h (queue included). This later evidence weakens earlier U/U4 timing
   forecasts. Their 8.8-hour guard and Kaggle's 9-hour limit remain material risks;
   no completion estimate in the historical reviews is a guarantee (D-B-280).

Evaluation limits remain: Gold58 is exposed; report-derived rulers share errors;
many decisions reuse the same holdout; ordinary OOF does not validate a fitted
nested stacker; additional seeds/folds can overlap the evaluation rows; and a
rounded public display or automatic selection order cannot establish private merit.
The independent result check in D-B-278 is distinct from a completed different-host
discovery cross-check. Latest entries with noncanonical or pending review fields
remain in that state. See D-A-002, D-B-016/160/171/248/259/278.

D-B-265 records unresolved host-ruling questions about external-checkpoint sources
as of its dated forum read. This report makes no new ruling or eligibility claim.

## 6. Existing final-pair rule and tomorrow's record

The governing policy is [plan.md, Final 10-slot schedule](../../plan.md), especially
Updates **7-10**, fixed before U/P4/U4/F4 outcomes. This report preserves it:

- Default remains **X + F**. New candidates require COMPLETE plus a displayed
  score by **10-23 08:00 KST**; pending/error candidates do not displace the pair.
- U/P4/U4 can replace A through their registered result branches. For the explicit
  `>=0.959` tie case, preference is **P4 > U4 > U**, reflecting fewer simultaneous
  changes and the existing evidence. A 0.958 automatic-selection branch in Updates
  7/8 is a recommendation for Taeyang's decision, not an automatic final lock.
- Update 10 treats displayed differences below 0.002 as ties. Do not silently
  resolve boundary cases by the older generic automatic-selection rule.
- If A is P4 or U4 and F4 is scored at least 0.957, B becomes F4. Otherwise B is F.
  F4 does not replace F when A is X. A5/DF are excluded under their registered rule.
- The final pair must be selected manually and its exact refs/versions confirmed.
  No such confirmation is present at this cutoff. A prepared F4 is not a submission.

When Taeyang provides tomorrow's results, record each exact ref, displayed score,
completion/error status and reported observation time; distinguish public results
from private score/rank. Apply the existing rule without retuning weights, confirm
the final two selected refs, and append the private result when available. Preserve
this pre-result snapshot so the later outcome cannot rewrite the original reasoning.
The user's expected result date is not authorization for a new score watcher.

## 7. Remaining records and safe resumption

The unified queue currently contains 11 literal item blocks, several with older
descriptions that were superseded by later findings. No claims were reassigned or
retired by this documentation task. The table below is a map, not a launch queue:

| Existing items | Later evidence / remaining dependency |
|---|---|
| H219 / H225 | Shared-render and drop-O submissions already have D-B-249/260 results; final-day outcomes belong to their newer refs |
| H223 / H226 | Extra-seed/backbone reserves; no new run inferred from the old claims |
| H224 / H231 | Fold/all-data line has C's result (D-B-263); no replay of completed fits |
| H227 | Per-target vector outcome recorded (D-B-251), not an untested proposal |
| H228 / H230 | R6 stride and runtime work advanced through D-B-250/253/261/262; do not rebuild v8 from the old item text |
| H35 / H107 | Exact final selection and its material runtime/failure evidence still matter; use the latest policy and exact selected versions |

The consistency checker reports **800 existing findings** at the start of this
documentation task. These include unified-queue format and historical metadata
debt; a clean project-wide PASS is not claimed. The current catalog records source
review labels without repairing scientific ownership or inventing verdicts.
The discovery-index Markdown fence was repaired for readability only.

The GitHub snapshot contains the four coordination documents, CLAUDE.md,
allowlisted documentation and Python/Markdown tools, evidence/attempt ledgers,
this report and preserved aggregate reviews. Data, labels, radiology reports,
weights, caches, credentials and private scratch stay outside publication.
The copy-only v1.0.3 publisher retains old files, applies the existing sanitizer,
checks the broad v1.0.2 secret pattern set and the 20 MiB file limit, and verifies
the exact private remote. Workspace coordination Git sync remains off.

Presentation update, 2026-10-22: English remains the main language; the Korean
companion and four original data-free figures explain the same scientific cutoff.
Publisher v1.0.4 adds these exact media paths and the Korean README, preserving
the above guards and checking PNG hashes/metadata. See the [visual sources and
reproduction guide](visuals-20261022/README.md). No patient imagery is published.

Some source evidence paths point into local-only scratch/artifacts and are not
downloadable from the notes repository. This is a documented research snapshot,
not a self-contained data/checkpoint release or a promise of full remote reproduction.

Operational follow-up, 2026-10-22 18:55 KST: the user's external-storage cleanup
removed 863.81 GiB of confirmed obsolete data while retaining current work, final
artifacts and specified cache copies. The [cleanup result](external-cleanup-result-20261022.md)
records exact scope, preserved evidence and remaining conditional archives. This
does not change the scientific findings or final-selection policy above.
