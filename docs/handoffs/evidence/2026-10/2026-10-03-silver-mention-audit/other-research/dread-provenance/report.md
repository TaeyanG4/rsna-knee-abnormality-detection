# Dread supervision provenance: source-only check

Observed 2026-10-03 KST. Scope is exactly the two named stored TXT sources and dataset metadata. Actual CSV/parquet values, reports, Gold, predictions, notebook outputs and blind packet were not accessed; no model/notebook execution. [Exact snapshots and hashes](manifest.json), [quoted source](source-excerpts.txt). Local snapshot prefixes are not verified Kaggle version IDs.

**Conclusion:** Dread's metadata claims calibrated probabilities and better training with soft labels, but the inspected material does not disclose the generation prompt, named LLM, calibration protocol or independent assessment needed to verify those claims. This is an evidence limitation, not a finding that the claims are false. Qwen/Flight generation rules must not be attributed to Dread.

| Question | What the specified sources show | Evidence limit |
|---|---|---|
| Label origin | Metadata `/info/description`: LLM parses reports into 12 probabilities. Training L29–36 and inference L20–24 describe suspected tear as approximately .8. | No prompt/template, model/version, decoding settings, extraction code or per-target uncertainty/negation contract disclosed. The .8 statement is an illustrative policy, not a measured calibration result. |
| Calibrated probability | Metadata uses those words. | No calibration method, separate calibration set, reliability curve, Brier/ECE/log-loss results or before/after comparison located. Raw probability range alone establishes no calibration. |
| Soft labels outperform thresholded labels | Metadata explicitly asserts this; training L164–165 repeats the preference. | No controlled soft/hard comparison specification, chosen threshold, matched split/seed/budget, table of results or uncertainty estimate is provided in these three sources. |
| Independent label review | Source describes withholding official labeled studies from training and image-model validation. | No separate blinded report-label adjudication, reader agreement, target-level label accuracy or extraction calibration validation is documented. Student AUC claims are not direct proof of calibrated report probabilities. |
| Uncertainty masking | Training uses external 12-dimensional target vectors directly. | No per-target uncertainty/unknown mask or confidence field in the inspected dataset-to-loss path. Input slice masks serve image sampling only. |

## Implemented training path

The training TXT says labels are supplied externally (L20–21, L35–36). L398–405 chooses `--labels` or `labels_llm_soft.parquet`; without it, the notebook exits successfully with a message. Therefore a successful notebook status would not by itself prove training occurred. The metadata separately advertises CSV/parquet availability; this apparent documentation-time difference does not identify the exact historical generator or run input.

L407 excludes official labeled IDs from `train_ids`; L419–421 loads the twelve columns as float32. `StudyWindows` returns `(x,y)` at L267–268, and collation returns only images and targets (L271–274). L425–426 derives class positive weights from mean training soft targets, clipped to `[1,10]`. L491 constructs `nn.BCEWithLogitsLoss(pos_weight=pw)` and L514–516 applies it directly. No uncertainty state, missing-label exclusion or confidence multiplier is passed to this loss. These are class-balance weights, unlike the earlier Pilkwang/Flight confidence weights. Soft targets preserve graded values but do not implement unknown masking.

The official-label dataset supplies `vds` (L448), and validation AUC selects checkpoints (L520–526). This confirms an implemented development-selection path; it is not separate extraction calibration or an untouched final assessment. Actual execution and historical score assertions were not verified.

## Source identity

- Training TXT `8-6ea209b03c33bf21.txt`: SHA256 `9f7055ef5dbec5227bd904290690f97f2aaa6447167c7d800acd961bf61f1045`.
- Inference TXT `7-859aae12ecea593b.txt`: SHA256 `9a86329225c8f975967bfa563fadc9ed06ce9d44c83c232439b0f5a1116a623e`.
- Dataset metadata, dataset ID 11640568: SHA256 `494bd44c457f8c3c91ce4813750ec233d9eba3be42cd0c56d24981f858f74778`.

Treat Dread as a separate, incompletely documented source of report-derived soft supervision. The inspected snippets do not establish the current local training script's behavior, the CSV-to-parquet identity, or the quality of individual labels. Future provenance requests should ask for the original prompt/model/version, uncertainty and silence policy, any calibration data/procedure, and the controlled soft/hard comparison. No relabeling or retraining is justified solely by missing documentation.
