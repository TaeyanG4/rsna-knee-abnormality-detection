# Bend the knee to the dinosaurs — a knee MRI ensemble, taken apart

**Public score 0.941.** Same weights, same arithmetic, same attribution as the
release it is built from. What changed is that you can now read it.

## What the score rewards, and why it shapes everything below

Macro ROC-AUC over twelve binary findings. Two consequences worth carrying
through the whole notebook:

- **Only ordering counts, per column.** Any monotone transform is free and
  invisible. Several appear below and none of them changes a score by itself —
  they matter only where they feed a blend.
- **Ties are expensive.** A study nobody could score gets a constant, and a
  block of identical constants earns half credit against every study it ties
  with. The EDA section measures how close this data comes to that failure.
- **Every finding counts equally.** A finding positive in 4% of studies is
  worth exactly as much as one positive in 40%.

## The pipeline

| Stage | Model | Contribution |
|---|---|---|
| 1 | DINOv2/v3 slot attention, multi-member | base ordering |
| 2 | A5 attention-pooling folds | 45% against stage 1 |
| 3 | RadImageNet ResNet-50 + heads, twin E10 / E13 / E11 layouts | 0.50 reference, 0.15 second pass, 0.40 calibrator |
| 4 | CoAtNet: four Raptor views, plus private resgated e4/e6/e8 at 0.40 | 0.60–1.00 of the final blend, per finding |

Stage 1 also ships a hand-written multilingual report extractor. It only runs
when the released weight package is absent, so in a normal Kaggle run it is
inert — but it is the most interesting code in the notebook, because it is what
produced most of the labels these weights were fitted on.

## What changed in this build

**One config cell.** Eleven fusion weights, previously scattered across four
stages as literals — one of them written `0.40000000000000002`, another
duplicated into an assertion 40 lines later — now live in a single `RUN` dict.
The stages read from it, and the drift guards still fire if code and config
disagree.

**An EDA section.** Four cells on what a study contains, how the twelve
findings are distributed and co-occur, what the underlying reports look like
(including which languages a monolingual extractor would silently drop), and
why every crop in this pipeline is measured in millimetres rather than pixels.

**Diagnostics at the end.** Per finding: how much the two arms actually
disagree, and how large the biggest tie block is. Together they say whether the
outer weights are doing anything at all.

**Three presets.** `probe22` is the 0.941 submission. `parent` is the flat 0.60
routing it came from (0.939). `halfway` sits between them. The config cell
explains the trade.


## 0. Run configuration

## Exploratory analysis

Read-only, a couple of minutes, no effect on the submission. Skip to stage 1 if you only want the pipeline.

## Stage 1 — the report extractor

Inert when the weight package is attached. Included because it is what labelled the training data: sentence splitting, negation and laterality windows, and finding vocabulary across several languages.

## Stage 1 (continued) — DINO: read the study, combine the members

## Stage 2 — A5: attention-pooling folds at 45%

## Stage 3 — RadImageNet: reference, twin E10 and E13

## Stage 4 — CoAtNet: Raptor views, residual checkpoints, final routing

## Diagnostics — is the routing doing anything?