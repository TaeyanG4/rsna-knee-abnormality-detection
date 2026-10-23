# Final results — RSNA Knee Abnormality Detection

**English main** · [한국어](final-results.ko.md) · [Pre-result snapshot](project-summary-20261022.md)

**Final result recorded**

Prepared: 2026-10-22T19:48:36+09:00. Latest verified result update: 2026-10-23T11:22:26+09:00

The expected date, 2026-10-23, was supplied by Taeyang; release time is unconfirmed. This page accepts manually supplied results and does not poll scores.

## 1. Competition outcome

| Field | Recorded value |
| --- | --- |
| Result state | Final result recorded |
| Private score | 0.943 |
| Rank | 312 |
| Total teams | 6425 |
| Award / medal | Silver medal (top 5 %; silver cut-off rank 321) |
| Observed at (KST) | 2026-10-23T10:59:57+09:00 |
| Evidence / source | Kaggle private leaderboard via kaggle CLI (312 / 6,425) and Taeyang's screenshot of the leaderboard row |

A dash means unrecorded, never zero or failure. Overall competition score/rank is separate from individual submission scores; it is not calculated as an average of the final pair.

## 2. Confirmed final submissions

Selection state: **confirmed**. The pre-result default X + F is a policy fallback, not confirmation of actual selection.

| Slot | Candidate | Submission ref | Remote version |
| --- | --- | --- | --- |
| 1 | X | 57415972 | 1 |
| 2 | F | 57475409 | 1 |

Selection confirmation time / evidence: 2026-10-23T00:30:00+09:00 / Taeyang's screenshot of My Submissions with both entries checked (received about 00:30 KST); plan.md Update 12

Use the [policy frozen before outcomes](https://github.com/TaeyanG4/rsna-knee-abnormality-detection/blob/4172ccf879887e3fa95f00cb714214e17a2805dd/plan.md), Updates 7–10. Eligibility at 2026-10-23 08:00 KST and the recorded 08:59 close are separate from result publication. This page does not choose or change the pair.

## 3. Submission results

Baseline public displays are frozen at **2026-10-22 18:15 KST**. New observations occupy separate columns. Acceptance is not proof of hidden completion.

| Candidate | Ref / v | Public at cutoff | New inference status | New public | Private |
| --- | --- | --- | --- | --- | --- |
| X | 57415972 / v1 | 0.958 | Complete | 0.958 | 0.943 |
| p25 | 57427331 / v1 | 0.958 | Complete | 0.958 | 0.943 |
| F | 57475409 / v1 | 0.957 | Complete | 0.957 | 0.943 |
| A5 | 57484279 / v1 | 0.956 | Complete | 0.956 | 0.941 |
| DF | 57500332 / v1 | 0.955 | Complete | 0.955 | 0.940 |
| U | 57510427 / v1 | — | Error | — | — |
| P4 | 57512878 / v1 | — | Complete | 0.958 | 0.943 |
| U4 | 57514281 / v1 | — | Error | — | — |
| F4 | 57526613 / v1 | — | Error | — | — |

F4 was preparation-only at cutoff. Fill its exact accepted kernel/ref/version from a receipt before adding results. Recovery submissions get separate rows; historical identities are not overwritten.

### Exact identities and evidence

| Candidate | Kernel | Accepted at (KST) | Baseline evidence |
| --- | --- | --- | --- |
| X | taeyangg4/knee-lb-r6w958k1x3sr20-v2 | 2026-10-20T10:04:32+09:00 | D-B-249/260/276 |
| p25 | taeyangg4/knee-lb-r6w958k1n3s2p25-v1 | 2026-10-20T16:12:20+09:00 | D-B-260/276 |
| F | taeyangg4/knee-lb-r6w958k1n3s2p25flatpv10-v1 | 2026-10-21T18:23:00+09:00 | D-B-274 |
| A5 | taeyangg4/knee-lb-r6w958k1n3s2p25a5v10-v1 | 2026-10-21T22:40:30+09:00 | D-B-275/276 |
| DF | taeyangg4/knee-lb-r6w958k1n3s2p25a5flatpv10b-v1 | 2026-10-22T08:40:49+09:00 | D-B-280/281 |
| U | taeyangg4/knee-lb-r6w958k1uxpv10-v1 | 2026-10-22T13:58:57+09:00 | handoff 2026-10-22 13:59; acceptance ledger |
| P4 | taeyangg4/knee-lb-r6w958k1n4fbp25v10-v1 | 2026-10-22T15:20:09+09:00 | D-B-278; acceptance ledger |
| U4 | taeyangg4/knee-lb-r6w958k1u4fbv10-v1 | 2026-10-22T15:59:21+09:00 | handoff 2026-10-22 15:59; acceptance ledger |
| F4 | taeyangg4/knee-lb-r6w958k1n4fbp25flatpv10-v1 | 2026-10-22T22:39:20+09:00 | plan.md Update 10; preparation only at baseline cutoff |

| Candidate | Observed at (KST) | New result evidence |
| --- | --- | --- |
| X | 2026-10-23T10:59:57+09:00 | kaggle competitions submissions (team account), checked by B 2026-10-23 10:59 KST |
| p25 | 2026-10-23T10:59:57+09:00 | kaggle competitions submissions (team account), checked by B 2026-10-23 10:59 KST |
| F | 2026-10-23T10:59:57+09:00 | kaggle competitions submissions (team account), checked by B 2026-10-23 10:59 KST |
| A5 | 2026-10-23T10:59:57+09:00 | kaggle competitions submissions (team account), checked by B 2026-10-23 10:59 KST |
| DF | 2026-10-23T10:59:57+09:00 | kaggle competitions submissions (team account), checked by B 2026-10-23 10:59 KST |
| U | 2026-10-23T10:59:57+09:00 | kaggle competitions submissions (team account), checked by B 2026-10-23 10:59 KST; COMPLETE without a score after &lt;= 9.18 h (hidden run failed, D-B-282) |
| P4 | 2026-10-23T10:59:57+09:00 | kaggle competitions submissions (team account), checked by B 2026-10-23 10:59 KST |
| U4 | 2026-10-23T10:59:57+09:00 | kaggle competitions submissions (team account), checked by B 2026-10-23 10:59 KST; COMPLETE without a score after &lt;= 9.18 h (hidden run failed, D-B-282) |
| F4 | 2026-10-23T10:59:57+09:00 | kaggle competitions submissions (team account), checked by B 2026-10-23 10:59 KST; COMPLETE without a score after &lt;= 9.21 h (hidden run failed, D-B-282) |

## 4. Interpretation after results

- Both finals scored private 0.943; we were the first of the 304 teams displaying 0.943, one display step above the bronze cut-off (0.942). (private leaderboard (kaggle CLI); D-B-282)
- No submission of ours scored above private 0.943; 21 of the 87 scored pipeline submissions reached it. (docs/data/submissions.csv)
- Three of the four Oct 22 afternoon/evening submissions (U, U4, F4) failed on hidden data after about 9.2 h; P4 completed in &lt;= 8.24 h. (D-B-280, D-B-282)

These are descriptive comparisons, not controlled causal estimates: runtime/stride differences and rounded displays may matter. Keep public, private and exposed local evidence distinct; preserve negative findings and review states.

## 5. Update log

| Time (KST) | Source | Change |
| --- | --- | --- |
| 2026-10-23T10:59:57+09:00 | B (Claude Code) from the Kaggle CLI and Taeyang screenshots | Recorded the final private result (0.943, rank 312 / 6,425, silver), the confirmed final pair X + F, all late submission outcomes and F4. |
| 2026-10-23T11:22:26+09:00 | B (Claude Code), repository maintenance | Snapshot links now point to commit 4172ccf of the renamed repository (TaeyanG4/rsna-knee-abnormality-detection). On 2026-10-23 the history was squashed into one commit, which removed the pre-result commit 9e1509f; Updates 7-10 of plan.md and the report text are unchanged except one redacted owner quote and a two-line pointer to this page. |

## 6. Update this page

Edit [final-results-data.json](final-results-data.json) once, then run:

```powershell
python tools/render_final_results.py
python tools/render_final_results.py --check
```

Both languages regenerate from one record. See the [update guide](final-results-update-guide.md) for fields, evidence and guarded publication.

[Immutable pre-result report](https://github.com/TaeyanG4/rsna-knee-abnormality-detection/blob/4172ccf879887e3fa95f00cb714214e17a2805dd/docs/reports/project-summary-20261022.md) · [Documentation audit](documentation-audit-20261022.md)

<!-- Generated by tools/render_final_results.py; edit final-results-data.json instead. -->
