# RSNA Knee Plateau Reassessment - 2026-09-13

## Observed public LB
- Anchor control: 0.941
- EXP018 ACL/MCL inner CoAt 0.20: 0.941
- EXP031 ACL/MCL inner CoAt 0.00: 0.940
- EXP034 outer push: 0.941
- EXP036 Raptor v8 reverse diversity: 0.941
- EXP035 outer shrink: pending at time of reassessment
- Current leader observed via Kaggle CLI: 0.956

## Conclusion
The 0.941 family is on a highly correlated prediction manifold. Weight/routing perturbations are too small to create the ~+0.015 macro AUC needed for first place. Rank blending and final reranking make many weight changes near-no-ops unless pairwise study ordering crosses.

## Validation lessons
- Gold58 must not be used for repeated hyperparameter selection.
- EXP031 looked favorable in Gold routing diagnostics but dropped public LB to 0.940.
- Global canonicalization looked positive on Gold but paired 4345-study pseudo-OOF was negative (-0.00119 macro; CI mostly negative).
- Use report-grouped folds for model selection; Gold only once after fixed training.

## New priorities
1. Freeze 0.941 public ensemble as anchor. Stop broad inner/outer routing search.
2. Build independent clean 5-fold image models from external pretrained weights, not public comp-trained checkpoints.
3. Use adjacent 2.5D windows at 288px instead of sparse spread slices for the first clean baseline.
4. Improve weak supervision via confidence masking / target-specific unknown handling; avoid Gold-fitted correction models.
5. Add target specialists only if complementarity reproduces on grouped weak OOF.
6. OrthoDiffusion is not a standalone model, but small target-selective blending remains viable: nested paired Raptor+Ortho delta +0.00165 macro on 4345-study diagnostic; exclude/limit Fracture.

## Active work
- EXP034 adjacent center9 4-slot 288 cache: Kaggle CPU v2 running.
- New clean training harness: src/train_adjacent_2p5d_clean.py
  - report-grouped split from EXP026
  - Gold/quarantine excluded from training
  - 3 adjacent triplets per slot, 4 slots = 12 instances/study
  - end-to-end pretrained backbone fine-tuning
  - weak high-confidence OOF for epoch diagnostics
  - Gold evaluated only once after fixed training
- First A/B after cache download: DINOv2-S vs a CNN/ConvNeXt family under identical split/training budget.
