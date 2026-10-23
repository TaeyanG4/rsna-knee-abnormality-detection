# Documentation audit and result readiness — 2026-10-22

[English results](final-results.md) · [한국어 결과](final-results.ko.md) · [Pre-result report](project-summary-20261022.md)

This is a documentation and publication audit by Codex (slot A), not a new
independent scientific review. English remains the main language. Existing
experiment verdicts, active B work and the 18:15 KST scientific cutoff are preserved.

## Findings and corrections

| Finding | Correction |
|---|---|
| Main README still referred to publisher v1.0.3 after image/Korean support moved to v1.0.4 | Updated the current operational instruction to v1.0.4; retained the dated historical explanation in the report |
| A date-only README headline could make the snapshot look current throughout the day | Added the exact 18:15 KST cutoff and separated the snapshot from later result updates |
| Result date could be mistaken for an official release time | Identified October 23 as user-supplied expectation; release time remains unconfirmed |
| Two independently edited result translations could drift | Added one manually maintained JSON record and a deterministic English/Korean renderer |
| No structured place to record actual final selection, private score and rank | Prepared separate competition, confirmed-pair, submission-result, evidence and interpretation sections |
| Baseline public scores could be confused with new/private outcomes | Kept baseline displays in a frozen column and all new results as null until supplied |
| Mutable links could obscure the policy known before results | Linked the pre-result report and policy at immutable commit 9e1509f78d43cba7f1d03d8e7c0bbeecf549b00a |

한국어: 오래된 게시 도구 안내와 기준 시점 표시를 보완했습니다. 결과 페이지에는 공개·비공개,
완료 상태·선택 확인·대회 순위를 분리했고, 하나의 입력에서 양언어가 함께 갱신되도록 했습니다.

## Evidence checked

- Reconciled all eight accepted submission refs, kernels, remote versions and KST
  acceptance timestamps with the immutable local attempt ledger. All matched.
  Acceptance-time PENDING is not promoted to a live completion state.
- Compared English/Korean candidate formulas, K3/K4 membership, X versus X_u
  stride distinction, five reported public displays and final-pair conditions.
  No new numerical discrepancy was identified in this scope.
- Checked the full-bag aggregate against D-B-264/278: replacement mean approximately
  -0.0045; additional-member mean +0.001449016, rendered +0.001449. Kept different
  controls and exposed local-evaluation limitations explicit.
- Reused the prior visual inspection after confirming all four PNG hashes are
  unchanged. Pixel dimensions, PNG format and absence of text/EXIF metadata pass.
  They are original diagrams/aggregate charts, not patient MRI.
- Parsed publication Markdown for local links, image alt text and heading anchors;
  validated generated English/Korean results against the same JSON record.
- Exercised 17 focused renderer cases: pending/partial/final states, identical
  bilingual score precision, missing evidence/time, invalid scores, changed
  baseline, unaccepted F4 results, invalid chronology, missing change log,
  duplicate pair/ref, and invalid rank/team combinations. All passed. Synthetic
  test values remained local and never entered published result data.

## Scope and remaining limits

The result record has **nine candidate rows**: eight accepted identities and one
F4 placeholder. It contains **no new public/private score**, no asserted final
rank, no medal and no confirmed final pair. No score query, new automation,
submission, GPU work or dataset inspection was performed.

Known project-wide metadata/unified-queue consistency debt remains. This audit
does not claim a clean research-project checker or re-review every historical
discovery. The original 390-title catalog retains its source review states.
GitHub remains a documentation snapshot, not a dataset/checkpoint backup.

The chart renderer uses Malgun Gothic on this Windows host. Reproduction on
another platform requires a Hangul-capable font for equivalent appearance.
The result renderer itself uses only the Python standard library.

Local QA evidence: `.local/scratch/slot-a/results-readiness-20261022/`.
Publication uses the existing v1.0.4 allowlist, broad secret scan, 20 MiB guard
and exact private-remote checks. Exact remote SHA and browser verification are
recorded in the local publication receipt after upload.
