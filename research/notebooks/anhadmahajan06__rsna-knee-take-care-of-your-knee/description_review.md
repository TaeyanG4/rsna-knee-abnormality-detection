# RSNA Knee Abnormality Detection: Master Multi-Arm Foundation Pipeline (0.936 SOTA)

### Proven 0.936 Leaderboard Architecture

This production notebook unifies **four distinct visual foundation families** into a single calibrated inference pipeline:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FOUR-ARM MULTI-PLANAR FOUNDATION MATRIX                              │
├─────────────────────┬───────────────────────┬──────────────────────────────┬───────────────────────────┤
│ Foundation Arm      │ Resolution & Geometry │ Primary Target Advantage     │ Optimal Weighting Policy  │
├─────────────────────┼───────────────────────┼──────────────────────────────┼───────────────────────────┤
│ 1. DINOv2 ViT-Small │ 6 slots, 12 slices    │ Baker's Cyst, Fracture       │ Baseline Anchor (0.899)   │
│ 2. Cross-Series A5  │ 6 slots, 16 slices    │ ACL, MCL, Marrow Contusion   │ W = 0.45 (0.910)          │
│ 3. RadImageNet R50  │ 4 slots dual-layout   │ Degenerative OA, Effusion    │ 88-Feat Calibrator (0.920)│
│ 4. CoAtNet Raptor   │ 5 slots, 64-slice vol │ Meniscus (Lat/Med), Fractures│ Dual-GPU SWA (0.936)      │
└─────────────────────┴───────────────────────┴──────────────────────────────┴───────────────────────────┘
```


## Stage 1: DINOv2 ViT-Small Foundation Ensemble (Baseline 0.899)
- Evaluates 20 fingerprint-verified DINOv2 ViT-Small members across 10 sliding windows per slot.
- Caches 12 slices per slot at 336px with 130mm crop and natural DICOM slice ordering.


## Stage 2: Cross-Series Spatial Attention Arm A5 (Score: 0.910)
- Evaluates 5-fold cross-series spatial attention (`vit_small_patch16_dinov3.lvd1689m`) across 16 slices per slot.
- Blends with Stage 1 using verified rank weighting ($W=0.45$).


## Stage 3: Dual-Layout RadImageNet ResNet-50 Arm & V18 Calibration (Score: 0.920)
- Extracts 2048-dim GAP features from RadImageNet ResNet-50.
- Evaluates dual pulse sequence layouts: Layout 1 (E13: Sag-FS, Cor-FS, Ax-FS, Sag-noFS) + Layout 2 (E11: Sag-noFS, Cor-noFS, Ax-noFS, Sag-FS).
- Applies 88-feature protocol-aware transformer linear calibration to correct scanner metadata variation.
- Preserves focal gating: $\alpha_{\text{Baker's}} = 0.0$ and $\alpha_{\text{Fracture}} = 0.0$.


## Stage 4 & 5: CoAtNet Raptor Arm & DINOsaur V4.2 Dual-Checkpoint Target Fusion (Score: 0.936)
- Evaluates `coatnet_rmlp_2_rw_384` hybrid on 64-slice multi-planar volume across 5 anatomical slots.
- Dual-GPU execution: Primary `coatnet_v5_swa` on GPU 0 ($w=0.89$, $K=64$) + complement `coatnet_v4` on GPU 1 ($w=0.11$, $K=42$).
- Stage 5 applies correlation-aware percentile rank fusion with proven target weights:
  `Lateral Meniscus: 0.61, Fracture: 0.61, Medial Meniscus: 0.56, Lateral OA: 0.55, ACL: 0.53`.
