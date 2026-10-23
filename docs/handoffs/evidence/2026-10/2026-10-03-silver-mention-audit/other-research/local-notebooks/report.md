# Public notebook source audit: uncertainty, severity, and executed-path limits

Observed 2026-10-03 KST. Source-only review of two local snapshots; no notebook execution, outputs, actual report/label/Gold/prediction values, or GPU work. Cell numbers below are **zero-based**; line numbers are one-based within each cell. [Manifest](manifest.json), [quoted source](source-excerpts.txt), [source differences](lineage-source.diff).

## Conclusion

The regex fallback implements **heuristic soft targets and confidence-weighted BCE**, not unknown masking. Uncertain-only evidence becomes a soft positive and remains supervised. PF OA is matched by anatomy and broad OA terminology; the official extent/depth threshold is not enforced. The two notebooks share this implementation and do not provide independent evidence for its quality. Their source alone cannot establish which label table or runtime branch produced any reported score.

## Exact propagation

Pilkwang cell 11 and Flight cell 12 have identical source SHA256 `545439108697be830e7172e7625bfe2fd0bbbf3661f3a3b6cd84c0d374e711d9`.

- L1–17: `_polarity()` tests uncertainty before negation and normality across the entire clause. `anchor_end` is unused. A clause with both uncertainty and negation can enter the uncertainty branch. This is a scope risk, not a measured error rate.
- L91–103: positive mention severity is 1.0 (high only), 0.45 (low only), otherwise 0.75. The comment calls these values ordered, not calibrated.
- L106–150: positive or uncertain mentions take precedence over negative counts. Uncertain mentions set `best >= 0.30`; their count is not returned. Score is `min(.95, .50 + .42*best + .03*min(n_pos,3))`; confidence is `min(1, .55+.15*n_pos)`. With only negatives, score/confidence depend on negative count; silence is `.28/.05`.
- L153–175: outputs include target, confidence, positive count and negative count, with no uncertainty state or target mask.
- L177–194: a positive global OA clause can populate all three compartments when positive/negative counts are zero; uncertain-only also satisfies that condition. Synovitis can inherit Effusion signal without increasing its confidence. These are coded heuristics, not verified clinical relationships.

Arithmetic below is before cross-target corrections and assumes no context penalty/bonus. It is not evaluation on actual studies.

| Clause evidence | Target Y | Confidence | Loss weight `.25+.75*c` | Validation `Y>.5` |
|---|---:|---:|---:|---:|
| Silence | .280 | .05 | .2875 | 0 |
| One explicit negative | .160 | .57 | .6775 | 0 |
| Uncertain only | .626 | .55 | .6625 | 1 |
| One low-severity positive | .719 | .70 | .7750 | 1 |
| One high-severity positive | .950 | .70 | .7750 | 1 |

Pilkwang cell 33 L81–90 / Flight cell 34 L81–90 builds targets with these confidence weights, while official annotated rows receive weight 3. Pilkwang L115 / Flight L172 thresholds soft labels at .5 for development AUC. Pilkwang L151–155 / Flight L208–212 applies `(BCEWithLogits(..., reduction="none") * w).mean()`.

Thus uncertainty and silence both contribute nonzero gradients in the regex path. `Mtr` is an image/slot availability mask, not a target uncertainty mask. Although the severity comment says absolute numbers do not matter to AUC, these numbers are also training targets and are thresholded into validation labels; their magnitudes can matter to both training and evaluation. No performance gain or loss can be inferred from this static audit.

## PF OA does not enforce the official definition

The [official definition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343), saved revision `92f17b192b0f7e5598d2d953db033726504d3387fa9e536516a6458462b2fa78`, observed 2026-10-02T02:10:53.896306+00:00, requires a moderate/large area (roughly 1 cm or greater) of high-grade cartilage loss (>50% thickness), with the same criterion for PF OA. It also describes borderline image findings as negative. [Definition-only excerpt](../../source-semantics/official-definition-excerpts.txt).

Pilkwang cell 8 L68–83 (Flight 9) accepts broad OA/chondropathy/cartilage-thinning/osteophyte terminology, combined with PF anatomy at L110–118. There is no joint extent-and-depth criterion. Cell 10 L1–11 (Flight 11) lists mild/small/focal as low severity, but one such positive still yields .719 in the unmodified branch. Global OA propagation further bypasses PF-specific severity evidence. This mismatch justifies a blinded source audit; it does **not** justify automatically converting unmentioned or ambiguous report findings into image-ground-truth negatives.

## Mounted labels and inference branch limit historical claims

Pilkwang cell 14 = Flight cell 15, SHA256 `871abcd8e68902415508f5eb40b3a05324a2fe63ef1bff2cd6850feb3e117f7a`:

1. `read_labels()` first derives regex labels (L109–112).
2. `find_label_table()` prefers a mounted `report_labels*.csv` or local `report_labels_v2.csv` (L54–85).
3. A usable mounted table overrides target and confidence columns for intersecting study IDs; uncovered IDs retain regex targets (L124–137). Missing expected columns or a declared label mount without a usable table raises an error.

Consequently this audit describes fallback semantics, not necessarily the selected training labels in a public run. Generation lineage and actual selected table require separate receipts. The inspected training source also retains official annotated rows in training if the split places them there; this must not be confused with our current receipt's Gold exclusion claim.

Both main functions (Pilkwang 33 / Flight 34 L30–42) return immediately through `infer_from_package()` when weights are found. Training is implemented, but the presence of training cells does not prove they ran in the scored notebook. No notebook outputs were used to resolve the branch.

## Source lineage

Pilkwang snapshot SHA256 `b32a9155fcc73c519e75e78c08e07d30393efc591c2798a3e40ca92f39832bb8`; Flight snapshot SHA256 `494df8db3b6f683b97648e7af0f79c098483100639ec42e741726c64203efe8b`.

After Flight's extra introductory cell, 32 of 35 inherited cells are exact source matches (18/21 code; all 14 inherited Markdown cells). All extractor cells and label loader match. Differences are configuration/cache/group choices (Pilkwang 13), appended per-target diagnostics (31), and training split/diagnostics (33). Source comments explicitly identify the fork. The pair is a shared implementation lineage, not two independent validations of uncertainty handling. Local snapshot filename integers are not verified Kaggle version IDs.

## EXP055 provenance-only receipt

`storage/labels/derived_EXP055/receipt.json`, SHA256 `1a935d2a52d50c9af770d1e8ba7c3fee5d6ed270222f064d42a587f6c825d6de`, claims Dread replacement for ACL, MCL, Effusion, Baker's and Fracture; other targets, including PF OA, remain Flight v4hybrid. Its confidence rule is `clip(2*abs(p-.5), .05, 1)`, which is a distance-from-.5 heuristic, not calibration evidence. Receipt claims Gold and quarantined duplicates are excluded from local training folds. CSV values, execution and coverage were not checked here.

## What to adopt next

Preserve explicit present/absent/uncertain/not-mentioned states, supporting text spans and source revision before proposing any target-specific mapping. Separate mention certainty from official severity. Audit cue scope, PF extent/depth and source-table coverage on the authorized blinded packet. Unknown masking or a changed soft-target policy is an experimental candidate requiring fixed train-only rules and comparison; these notebooks do not establish which choice improves the competition score. Do not retrain from these findings alone.
