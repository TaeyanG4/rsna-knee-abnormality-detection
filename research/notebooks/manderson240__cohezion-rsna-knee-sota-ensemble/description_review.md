# Bend the Knee to the Dinosaurs (ALL PUBLIC)

## New addition: CoAtNet MRI reader

Inspired by [dreaddevelopment's discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696), this version adds my public [CoAtNet MRI model](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coat-resgated-ep10-top3).

Each 384 px adjacent-slice triplet passes through a CoAtNet RMLP-2 backbone. A finding-specific gated spatial head blends localized evidence with a protected global-average residual; a second finding-specific attention stage pools windows across the study. Three checkpoints are combined by per-finding rank averaging before joining the existing ensemble.

Inference-only submission.

## Credits

[Pilkwang](https://www.kaggle.com/datasets/pilkwang/rsna-knee-weights) · Sofia Anjenje · Antoine G. · prvsiyan · Marwan Mahmoud · dreaddevelopment / Roman Tamrazov · [renta.k](https://www.kaggle.com/code/renta0426/rsna-knee-hybrid-raptor-lateral-meniscus-r100) · [Anvith Pothula](https://www.kaggle.com/code/anvithpothula/rsna-base)
