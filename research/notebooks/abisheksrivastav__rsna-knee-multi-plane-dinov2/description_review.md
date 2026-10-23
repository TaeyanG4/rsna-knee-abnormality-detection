# RSNA Knee: DINOv2 336px fold trainer V2

This notebook trains one fold and one seed per Kaggle run. It uses twelve slices from
six protocol-aware sequence slots, stores decoded studies in a compressed temporary
cache, fine-tunes DINOv2-small, and exports a checkpoint plus prediction artifacts.

Required Kaggle inputs:

- Competition: `rsna-knee-abnormality-detection`
- Model: `metaresearch/dinov2/PyTorch/small/1`
- Recommended labels: `pilkwang/rsna-knee-llm-labels`

Internet must remain disabled. Change `TRAIN_FOLD` and `RUN_SEED` between runs; do not
try to train a twenty-member ensemble in one nine-hour session.

## Configuration

The defaults target a Kaggle T4 notebook and leave a margin under the nine-hour limit.
Set `DEBUG=True` only to verify the pipeline; its submission is not competitive.

## Locate offline inputs

## Report-derived supervision

A mounted `report_labels*.csv` is preferred. The fallback below is intentionally
conservative: silence receives little weight, explicit negation receives medium weight,
and a named positive finding receives high weight. The 58 expert-labelled studies always
override report labels.

## DICOM ordering and multi-plane cache

File names are SOP UIDs and are not slice order. Each selected series is sorted using
`ImageOrientationPatient` and `ImagePositionPatient`, then sampled from the central 60%.
Images are cropped to a 130 mm field of view and normalized by robust percentiles. The
cache is compressed on temporary disk; the equivalent dense RAM array is too large at
336 px x 12 slices.

## DINOv2 study model

Each non-empty sequence slot becomes one three-slice RGB-like input. DINOv2 encodes the
slots; each abnormality has its own learned attention query over the six sequence slots.
Empty slots are excluded from both the encoder and the attention softmax.

## Targets and report-grouped folds

Duplicate reports remain in one fold. This prevents a templated report from creating the
same weak target in both training and validation.

## Train the selected fold and export artifacts

## Next competitive experiments

Train folds 0-4 under one seed before adding more seeds. Package the resulting
checkpoints as a private Kaggle Dataset and use an inference-only notebook to rank
average them. Keep the fold table fixed for every architecture.

1. Improve report labels for the weakest gold-agreement targets.
2. Add DINOv2-base or a supervised ConvNeXt as a genuinely different encoder.
3. Increase slice groups only when the runtime and RAM measurements support it.
4. Rank-average out-of-fold-validated models; do not choose blends from public LB noise.