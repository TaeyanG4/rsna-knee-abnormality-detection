# 🦕 DINOv2 Baseline + Knees in 9 Languages

**A 4-arm ensemble scoring 0.937+ on the RSNA Knee Abnormality Detection challenge.**

Based on [Dvorkin's RSNA Baseline](https://www.kaggle.com/code/evgendvorkin/rsna-baseline) — translated to English with added analysis.

## What's Inside

| Arm | Architecture | Models | Resolution | What It Does |
|-----|-------------|--------|------------|-------------|
| 🦕 **DINOv2** | ViT-small transformer | 20 cross-fold | 336×336 | Slot-based attention per diagnosis, 3 slices as RGB |
| 🦖 **DINOv3** | ViT-small volumetric | 5 cross-fold | 336×336 | 16 slices per slot, learns per-slice importance |
| 🏥 **RadImageNet** | ResNet50 (medical pretrain) | 10 heads | 224×224 | Domain-specific features from medical imaging |
| 🔬 **CoAtNet** | Conv+Attention hybrid | 4 variants | 384×384 | Strongest single arm, per-diagnosis window selection |

**Total: 39 models → rank-averaged ensemble → 0.937+ AUC**

### Added Analysis (not in the original)
- 📊 Language detection across 4,407 radiology reports (9 languages found)
- 🔍 Per-language NLP extraction quality breakdown
- 📈 How the multilingual lexicon handles negation in each language
- 🏗️ Clear English explanations of each ensemble arm


All the following experiments are also based on **publicly available works** in order to understand the logic of the changes made by top competitors. Pre-trained weights are loaded from public Kaggle datasets.

**Key references:**
- [Pilkwang Kim — RSNA Knee baseline v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) (526 votes)
- [Dread Development — Knee MRI: twelve findings](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model) (120 votes)
- [Mattia Angeli — Bend the Knee to DinoV3](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-dinov3-ensembled) (208 votes)


---
## 📊 Language Analysis: What Languages Are These Reports In?

Before diving into the ensemble, let's understand the data. The RSNA competition has **4,407 training studies** from 16 sites across 5 continents. The radiology reports are in **9 different languages** — and the NLP label extraction pipeline must handle all of them.


# 📊 Solution Comparison: 0.936 → 0.937–0.939

## Original Work (0.936 — Pilkwang Kim)

The pipeline consists of four arms:

1. **DINOv2 transformer** — 20 cross-fold models, 336×336
2. **DINOv3 volumetric** — 5 models with 16-slice xcodex pooling
3. **RadImageNet ResNet50** — 10 attention heads, medical pretraining
4. **CoAtNet** — conv+attention hybrid, strongest single arm

## What Changed (0.937–0.939)

### CoAtNet Improvements
The biggest gain came from the **CoAtNet arm**:
- Added a 4th sub-model (reverse horizontal flip variant)
- Updated from 3-arm to 4-arm CoAtNet blend with correlation-aware weighting
- Weights: maxspan-v5 (0.55) + native384dense-v10 (0.10) + maxspan-v5-reverse (0.15) + native384-v8 (0.20)

### Soft Labels from Reports (+0.013 AUC)
- Used LLM-parsed radiology reports to create **soft probability labels** (not hard 0/1)
- Expanded training from ~3,155 to 4,349 studies
- This single change was worth **+0.013 AUC** on gold validation

### Result
| Metric | Before | After |
|--------|--------|-------|
| Gold validation (58 samples) | 0.9211 | 0.9254 |
| Public LB | 0.936 | 0.937–0.939 |


# 🔄 Updated CoAtNet Arm: 4-Model Blend

## Previous Setup
The CoAtNet arm used one primary model (`maxspan-v5`) with SWA (Stochastic Weight Averaging).

## New Setup — 4 Sub-Models

| Model | Weight | Slices | Span | Windows | Notes |
|-------|--------|--------|------|---------|-------|
| **maxspan-v5** | 0.55 | 64 | 0.02–0.98 | 62 | Primary; full-span sampling |
| **native384dense-v10** | 0.10 | 64 | 0.02–0.98 | 62 | Different training schedule |
| **maxspan-v5-reverse** | 0.15 | 64 | 0.02–0.98 | 62 | **Horizontal flip** (free diversity) |
| **native384-v8** | 0.20 | 44 | 0.06–0.94 | 42 | Reduced span; coarser sampling |

**Why this works:** Each sub-model sees the data slightly differently. The reverse flip exploits anatomical near-symmetry. The coarser span model is less sensitive to edge slices. Blending diverse predictions reduces variance without introducing bias.

### Correlation-Aware Fusion
Instead of fixed weights across all 12 diagnoses, the ensemble checks **per-diagnosis correlation** between the transformer arms and CoAtNet:
- High correlation (r > 0.992): reduce CoAtNet weight (redundant signal)
- Low correlation (r < 0.60): increase CoAtNet weight (complementary signal)
- Base weight: 0.60 for CoAtNet, 0.40 for transformers


## 1. DINOv2 Transformer Arm

The first arm is the largest. It contains **20 DINOv2-small models** trained on different cross-validation folds. Each model:

- Takes 6 slots per study (SAG_FLUID_FS, COR_FLUID_FS, AX_FLUID_FS, SAG_FLUID_NOFS, COR_T1, SAG_T1)
- Stacks 3 adjacent slices as RGB channels (2.5D approach)
- Uses **SlotHead attention**: learns which anatomical plane matters for each diagnosis
- Resolution: 336×336 pixels

This is essentially [Pilkwang Kim's baseline](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) with 20-fold cross-validation.


## 2. DINOv3 Volumetric Arm

The second arm uses **5 DINOv3-small models**. Unlike the DINOv2 arm (3 slices per slot), this arm processes **16 slices per slot** — capturing 3D volumetric context.

**Key difference — xcodex pooling:**
- DINOv2: mean-pools 3 slices → loses depth information
- DINOv3: attention-based pooling over 16 slices → learns which slices matter per diagnosis
- Example: ACL tears are visible on specific sagittal slices; xcodex learns to attend to those

This arm provides **complementary signal** — pathologies that span multiple slices (cartilage damage, meniscus tears) benefit from the volumetric view.


## 3. RadImageNet ResNet50 Arm

The third arm uses **ResNet50 pre-trained on RadImageNet** (a medical imaging dataset of ~4.7M images), not ImageNet.

**Why medical pretraining matters:**
- ImageNet features: edges, textures, colors (natural photos)
- RadImageNet features: anatomy, tissue contrast, pathology patterns (medical images)

This arm uses:
- 3 anatomical planes × 8 slices = 24 inputs per study
- Multi-head attention (10 heads: 5 public + 5 E13 folds)
- Post-hoc calibration per diagnosis
- Resolution: 224×224


## 4. CoAtNet Attention Arm (Strongest Single Arm)

The fourth arm uses **CoAtNet** — a hybrid of convolutional layers (local features) + multi-head self-attention (global context). This is the **strongest individual arm** in the ensemble.

**Architecture:** `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` from timm
- **Input:** 64 slices from 5 slots, 384×384 resolution
- **RaptorClassifier:** Per-diagnosis attention over K windows (42-62 windows)
- **Each window:** 3 consecutive slices stacked as RGB

**Why CoAtNet is strongest:**
- Higher resolution (384 vs 336/224)
- More slices (64 vs 3-16)
- Hybrid architecture captures both local pathology and global anatomy
- Per-diagnosis window attention → focuses on the most informative slices for each condition


---
## Summary

This notebook runs a **4-arm, 39-model ensemble** achieving **0.937+ AUC** on the RSNA Knee Abnormality Detection challenge:

1. **DINOv2** (20 models) — slot-based attention, 3 slices as RGB
2. **DINOv3** (5 models) — volumetric, 16 slices with xcodex pooling
3. **RadImageNet** (10 heads) — medical domain pretraining
4. **CoAtNet** (4 sub-models) — strongest single arm, conv+attention hybrid

**The biggest insight:** Soft labels from LLM-parsed reports were worth +0.013 AUC alone. Better labels > bigger models.

**Language analysis added by this fork:**
- 9 languages detected across 4,407 reports
- 60%+ non-English — English-only NLP would miss the majority
- The multilingual lexicon handles all 9 languages simultaneously

---
**If this English translation and analysis was useful, please upvote!** Fork to build on this.

Credits: [Dvorkin](https://www.kaggle.com/code/evgendvorkin/rsna-baseline), [Pilkwang Kim](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1), [Dread Development](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model), [Tony Li](https://www.kaggle.com/code/tonylica/rsna-knee-dino-radimagenet-rank-ensemble)
