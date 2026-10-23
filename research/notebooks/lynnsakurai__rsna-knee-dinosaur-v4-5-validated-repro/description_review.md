# RSNA Knee — optimized multimodal ensemble

**Goal:** maximize the macro-average ROC AUC for the 12 knee findings while keeping the scored notebook offline, reproducible, and below the 9-hour GPU limit.

This is an **inference-only** notebook. The attached public checkpoints were trained beforehand; the commit only reads hidden-test DICOMs and writes `/kaggle/working/submission.csv`. Reports are training-only and are not assumed to exist for test studies.

The prediction graph is the strongest validated graph in the supplied examples:

1. protocol-aware DINO multi-view ensemble;
2. RadImageNet diversity branch and calibration;
3. efficient CoAtNet study-window branch;
4. an optional OrthoDiffusion specialist admitted only for **ACL** and **Effusion**;
5. fixed, target-wise percentile-rank fusion.

The source notebooks report a public score of **0.936** for the parent graph. That is historical evidence, not a guarantee for the private leaderboard. External assets remain subject to their original licenses. Research use only; not a medical device.

## Objective and why the blend is rank-based

For target (k), empirical AUC is the fraction of positive/negative pairs ordered correctly:

\[
\widehat{AUC}_k=\frac{1}{n_k^+n_k^-}
\sum_{i:y_{ik}=1}\sum_{j:y_{jk}=0}
\left[\mathbf{1}(s_{ik}>s_{jk})+\tfrac12\mathbf{1}(s_{ik}=s_{jk})\right].
\]

The competition score gives every finding equal weight:

\[
S=\frac{1}{12}\sum_{k=1}^{12}\widehat{AUC}_k.
\]

AUC depends on ordering, not absolute calibration. Each model output is therefore converted to a within-target percentile rank (R_m(s_{ik})), then blended with fixed OOF-selected weights:

\[
z_{ik}=\sum_m w_{mk}R_m(s^{(m)}_{ik}),\qquad
\hat p_{ik}=R(z_{ik}),\qquad \sum_m w_{mk}=1.
\]

This prevents one arm's logit scale from dominating. Weights are target-specific because the useful plane and architecture differ for ligaments, menisci, OA, fluid, and bone findings. Test labels, test prevalence, and public-leaderboard feedback are never used to fit them.

## Training strategy represented by the checkpoints

The high-value design choices are compact:

- **Use reports only as teachers.** Multilingual report-derived soft labels expand supervision beyond the small fully adjudicated subset. Gold image labels remain authoritative; ambiguous report labels receive lower weight. No report is needed at inference.
- **Split by study.** Keep all series from one `StudyInstanceUID` in one fold. Use iterative multilabel stratification on the adjudicated labels and audit near-duplicate studies before training.
- **Normalize in physical space.** Order slices from DICOM geometry, crop a fixed millimetre field of view, robust-window each series, and preserve sagittal/coronal/axial plus fluid-sensitive/fat-suppressed slots.
- **Use 2.5D multiple-instance learning.** Three adjacent slices form RGB-like input; target-specific attention pools slice/window evidence to study logits. This is much cheaper than full 3D training and handles variable series length.
- **Spend capacity on diversity.** DINO contributes strong global representation, RadImageNet contributes medical-image bias, CoAtNet supplies efficient local/global windows, and OrthoDiffusion is used only where OOF evidence justified it.

For retraining, minimize a masked mixture of gold BCE, confidence-weighted report distillation, and a small pairwise ranking term:

\[
\mathcal L=\mathcal L_{gold}+\lambda_p\sum_{ik}c_{ik}\,
\mathrm{BCE}(\sigma(s_{ik}),\tilde y_{ik})+
\lambda_r\sum_k\log\left(1+e^{-(s_k^+-s_k^-)}\right).
\]

Select epochs and blend weights exclusively on out-of-fold macro-AUC. Because the metric is macro-averaged, report all 12 validation AUCs—not only their mean.

## Runtime plan

- Kaggle GPU notebook, preferably **T4 ×2**; internet **off**.
- Keep every public model/dataset input recorded in this notebook's Kaggle metadata attached.
- DICOM decoding is cached and parallelized; model arms are released between stages.
- The parent ensemble is fail-safe: a specialist cannot erase a valid earlier submission.
- Do not add test-time calibration based on the three visible example studies—the hidden rerun contains the real test set.

## 1. OrthoDiffusion specialist inference

The official knee-MRI foundation is transferred plane-wise. Only ACL (5% rank weight) and effusion (50%) survived grouped validation; its other outputs are deliberately ignored. If the `orthodiffusion-fold1-knee-specialist` input is unavailable, this branch is skipped and the validated parent submission remains intact.

## 2. Validated parent ensemble

The following four cells are kept model-identical to the supplied validated reproduction. Their strict path, hash, schema, and fallback checks are intentional: changing preprocessing without retraining would invalidate the checkpoints.

## 3. Final submission audit

The last check is deliberately simple and strict. Kaggle only needs one artifact: `submission.csv` with the sample schema, hidden-test row order, finite scores, and values in ([0,1]).